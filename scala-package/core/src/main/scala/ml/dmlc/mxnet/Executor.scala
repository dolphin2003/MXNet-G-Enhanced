
package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ArrayBuffer

object Executor {
  // Get the dictionary given name and ndarray pairs.
  private[mxnet] def getDict(names: Seq[String],
                             ndarrays: Seq[NDArray]): Map[String, NDArray] = {
    require(names.toSet.size == names.length, "Duplicate names detected")
    (names zip ndarrays).toMap
  }

  /**
   * Get input slice from the input shape.
   * @param batchSize The number of samples in a mini-batch.
   * @param workLoadList The list of work load for different devices, in the same order as ctx
   * @return The split slices to get a specific slice.
   * @throws IllegalArgumentException
   *         If there are two many splits such that some slice can be empty.
   */
  private[mxnet] def splitInputSlice(batchSize: Int,
                                     workLoadList: Seq[Float]): Array[(Int, Int)] = {
    val totalWorkLoad = workLoadList.sum
    val batchNumList = workLoadList.map(workLoad =>
      math.round(workLoad * batchSize / totalWorkLoad)).toArray
    val batchNumSum = batchNumList.sum
    if (batchNumSum < batchSize) {
      batchNumList(batchNumList.length-1) += batchSize - batchNumSum
    }

    val slices = ArrayBuffer.empty[(Int, Int)]
    var end = 0
    batchNumList.foreach(batchNum => {
      val begin = math.min(end, batchSize)
      end = math.min(begin + batchNum, batchSize)
      require(begin < end, "Too many slices such that some splits are empty")
      slices.append((begin, end))
    })
    slices.toArray
  }

  /**
   * Check the argument names of symbol.
   * This function checks the duplication of arguments in Symbol.
   * The check is done for feedforward net for now.
   * @param symbol The network configuration
   */
  private[mxnet] def checkArguments(symbol: Symbol): Unit = {
    val argNames = symbol.listArguments()
    require(argNames.toSet.size == argNames.length,
      "Find duplicated argument name," +
      "please make the weight name non-duplicated(using name arguments)," +
      s"arguments are $argNames")

    val auxNames = symbol.listAuxiliaryStates()
    require(auxNames.toSet.size == auxNames.length,
      "Find duplicated auxiliary param name," +
      "please make the weight name non-duplicated(using name arguments)," +
      s"arguments are $auxNames")
  }

  // Load a list of arrays into a list of arrays
  private[mxnet] def loadGeneral(data: Seq[NDArray], targets: Seq[NDArray]): Unit = {
    (data zip targets).foreach { case (dSrc, dTarget) =>
      dSrc.copyTo(dTarget)
    }
  }

  // Load a list of arrays into a list of arrays specified by slices
  private[mxnet] def loadGeneralMulti(data: Seq[NDArray],
                                      targets: Seq[Array[(Int, Int, NDArray)]]): Unit = {
    for ((src, dTargets) <- data zip targets) {
      for ((start, end, dst) <- dTargets) {
        val sliced = src.slice(start, end)
        sliced.copyTo(dst)
        sliced.dispose()
      }
    }
  }

  // Load data into sliced arrays
  private[mxnet] def loadDataMulti(batch: DataBatch,
                                   targets: Seq[Array[(Int, Int, NDArray)]]): Unit = {
    loadGeneralMulti(batch.data, targets)
  }

  private[mxnet] def loadData(batch: DataBatch, targets: Seq[NDArray]): Unit = {
    loadGeneral(batch.data, targets)
  }

  // Load label into sliced arrays
  private[mxnet] def loadLabelMulti(batch: DataBatch,
                                    targets: Seq[Array[(Int, Int, NDArray)]]): Unit = {
    loadGeneralMulti(batch.label, targets)
  }

  private[mxnet] def loadLabel(batch: DataBatch, targets: Seq[NDArray]): Unit = {
    loadGeneral(batch.label, targets)
  }
}

/**
 * Symbolic Executor component of MXNet <br />
 * <b>
 * WARNING: it is your responsibility to clear this object through dispose().
 * NEVER rely on the GC strategy
 * </b>
 *
 * @author Yizhi Liu
 *
 * Constructor: please use Symbol.bind and Symbol.simpleBind instead.
 * @param handle ExecutorHandle generated by calling Bind
 * @param symbol
 * @see Symbol.bind : to create executor
 */
// scalastyle:off finalize
class Executor private[mxnet](private[mxnet] val handle: ExecutorHandle,
                              private[mxnet] val symbol: Symbol) {
  private[mxnet] var argArrays: Array[NDArray] = null
  private[mxnet] var gradArrays: Array[NDArray] = null
  private[mxnet] var auxArrays: Array[NDArray] = null
  val outputs: Array[NDArray] = getOutputs
  protected var _argDict: Map[String, NDArray] = null
  protected var _auxDict: Map[String, NDArray] = null
  protected var monitorCallback: MXMonitorCallback = null
  private[mxnet] var _ctx: Context = null
  private[mxnet] var _gradsReq: Iterable[_] = null
  private[mxnet] var _group2ctx: Map[String, Context] = null

  private var disposed = false

  override protected def finalize(): Unit = {
    dispose()
  }

  def dispose(): Unit = {
    if (!disposed) {
      outputs.foreach(_.dispose())
      _LIB.mxExecutorFree(handle)
      disposed = true
    }
  }

  /**
   * Return a new executor with the same symbol and shared memory,
   * but different input/output shapes.
   * For runtime reshaping, variable length sequences, etc.
   * The returned executor shares state with the current one,
   * and cannot be used in parallel with it.
   * @param partialShaping Whether to allow changing the shape of unspecified arguments.
   * @param allowUpSizing Whether to allow allocating new ndarrays that's larger than the original.
   * @param kwargs Map of string to Shape.
   *                - new shape for arguments.
   * @return
   * executor A new executor that shares memory with this.
   */
  def reshape(partialShaping: Boolean = false, allowUpSizing: Boolean = false,
    kwargs: Map[String, Shape]): Executor = {
     val (argShapes, _, auxShapes) = this.symbol.inferShape(kwargs)
     require(argShapes != null, "Insufficient argument shapes provided.")

    var newArgDict = Map[String, NDArray]()
    var newGradDict = Map[String, NDArray]()

    this.symbol.listArguments().zipWithIndex.foreach { case (name, i) =>
      val newShape = argShapes(i)
      val arr = this.argArrays(i)
      val dArr = if (this.gradArrays == null) null else this.gradArrays(i)
      if (partialShaping || kwargs.contains(name) || newShape.equals(arr.shape)) {
        if (newShape.product > arr.shape.product) {
          require(allowUpSizing, s"New shape of arg:$name larger than original. " +
                        "First making a big executor and then down sizing it " +
                        "is more efficient than the reverse." +
                        "If you really want to up size, set allowUpSizing = true " +
                        "to enable allocation of new arrays.")
          newArgDict = newArgDict + (name -> NDArray.empty(newShape, arr.context))
          if (dArr != null) {
            newGradDict = newGradDict + (name -> NDArray.empty(newShape, dArr.context))
          }
        } else {
          newArgDict = newArgDict + (name -> arr.reshape(newShape.toArray))
          if (dArr != null) {
            newGradDict = newGradDict + (name -> dArr.reshape(newShape.toArray))
          }
        }
      } else {
        import java.lang.AssertionError
        throw new  AssertionError(s"Shape of unspecified array arg:$name changed." +
                    "This can cause the new executor to not share parameters " +
                    "with the old one. Please check for error in network." +
                    "If this is intended, set partialShaping = true to suppress this warning.")
      }
    }

    var newAuxDict = Map[String, NDArray]()
    val zip3 = (this.symbol.listAuxiliaryStates, auxShapes, this.auxArrays).zipped
    zip3.foreach { case (name, newShape, arr) =>
      if (partialShaping || newShape.equals(arr.shape)) {
        if (newShape.product > arr.shape.product) {
          require(allowUpSizing, s"New shape of aux:$name larger than original. " +
                        "First making a big executor and then down sizing it " +
                        "is more efficient than the reverse." +
                        "If you really want to up size, set allowUpSizing = true " +
                        "to enable allocation of new arrays.")
          newAuxDict = newAuxDict + (name -> NDArray.empty(newShape, arr.context))
        } else {
          newAuxDict = newAuxDict + (name -> arr.reshape(newShape.toArray))
        }
      } else {
        import java.lang.AssertionError
        throw new  AssertionError(s"Shape of unspecified array aux:$name changed." +
                  "This can cause the new executor to not share parameters " +
                  "with the old one. Please check for error in network." +
                  "If this is intended, set partialShaping = true to suppress this warning.")
      }
    }
    if (this._gradsReq.isInstanceOf[Seq[_]]) {
      this.symbol.bind(this._ctx,
                          newArgDict,
                          newGradDict,
                          this._gradsReq.asInstanceOf[Seq[String]],
                          newAuxDict,
                          this._group2ctx,
                          this)
    } else {
      this.symbol.bind(this._ctx,
                          newArgDict,
                          newGradDict,
                          this._gradsReq.asInstanceOf[Map[String, String]],
                          newAuxDict,
                          this._group2ctx,
                          this)
    }
  }

  /**
   * list all the output ndarray
   * @return A list of ndarray binded to the heads of executor.
   */
  private def getOutputs: Array[NDArray] = {
    val ndHandles = ArrayBuffer[NDArrayHandle]()
    checkCall(_LIB.mxExecutorOutputs(handle, ndHandles))
    ndHandles.toArray.map(new NDArray(_))
  }

  /**
   * Calculate the outputs specified by the binded symbol.
   * @param isTrain whether this forward is for evaluation purpose.
   * @param kwargs Additional specification of input arguments.
   */
  def forward(isTrain: Boolean, kwargs: (String, NDArray)*): Unit = {
    kwargs.foreach { case (name, array) =>
      require(argDict.contains(name), s"Unknown argument $name")
      array.copyTo(argDict(name))
    }
    checkCall(_LIB.mxExecutorForward(handle, if (isTrain) 1 else 0))
  }

  def forward(): Unit = {
    forward(isTrain = false)
  }

  /**
   * Do backward pass to get the gradient of arguments.
   * @param outGrads Gradient on the outputs to be propagated back.
   *                 This parameter is only needed when bind is called
   *                 on outputs that are not a loss function.
   */
  def backward(outGrads: Array[NDArray]): Unit = {
    require(outGrads != null)
    val ndArrayPtrs = outGrads.map(_.handle)
    checkCall(_LIB.mxExecutorBackward(handle, ndArrayPtrs))
  }

  def backward(outGrad: NDArray): Unit = {
    require(outGrad != null)
    backward(Array(outGrad))
  }

  def backward(): Unit = {
    backward(Array.empty[NDArray])
  }

  /**
   * Install callback.
   * @param callback Takes a string and an NDArrayHandle.
   */
  def setMonitorCallback(callback: MXMonitorCallback): Unit = {
    monitorCallback = callback
    checkCall(_LIB.mxExecutorSetMonitorCallback(handle, monitorCallback))
  }

  /**
   * Get dictionary representation of argument arrrays.
   * @return The dictionary that maps name of arguments to NDArrays.
   * @throws IllegalArgumentException if there are duplicated names in the arguments.
   */
  def argDict: Map[String, NDArray] = {
    if (_argDict == null) {
      _argDict = Executor.getDict(symbol.listArguments(), argArrays)
    }
    _argDict
  }

  /**
   * Get dictionary representation of auxiliary states arrays.
   * @return The dictionary that maps name of auxiliary states to NDArrays.
   * @throws IllegalArgumentException if there are duplicated names in the auxiliary states.
   */
  def auxDict: Map[String, NDArray] = {
    if (_auxDict == null) {
      _auxDict = Executor.getDict(symbol.listAuxiliaryStates(), auxArrays)
    }
    _auxDict
  }

  /**
   * Copy parameters from arg_params, aux_params into executor's internal array.
   * @param argParams : dict of name to NDArray of arguments
   * @param auxParams : dict of name to NDArray of auxiliary states.
   * @param allowExtraParams
   *        Whether allow extra parameters that are not needed by symbol
   *        If this is True, no error will be thrown when arg_params or aux_params
   *        contain extra parameters that is not needed by the executor.
   * @throws IllegalArgumentException
   *         If there is additional parameters in the dict but allow_extra_params=False
   */
  def copyParamsFrom(argParams: Map[String, NDArray],
                     auxParams: Map[String, NDArray],
                     allowExtraParams: Boolean = false): Unit = {
    argParams.foreach { case (name, array) =>
      if (argDict.contains(name)) {
        array.copyTo(argDict(name))
      } else {
        require(allowExtraParams, s"Find name $name that is not in the arguments")
      }
    }
    if (auxParams != null) {
      auxParams.foreach { case (name, array) =>
        if (auxDict.contains(name)) {
          array.copyTo(auxDict(name))
        } else {
          require(allowExtraParams, s"Find name $name that is not in the auxiliary states")
        }
      }
    }
  }

  def copyParamsFrom(argParams: Map[String, NDArray], allowExtraParams: Boolean): Unit = {
    copyParamsFrom(argParams, null, allowExtraParams)
  }

  def copyParamsFrom(argParams: Map[String, NDArray]): Unit = {
    copyParamsFrom(argParams, allowExtraParams = false)
  }

  /**
   * Get a debug string about internal execution plan.
   * @return Debug string of the executor.
   */
  def debugStr: String = {
    val str = new RefString
    checkCall(_LIB.mxExecutorPrint(handle, str))
    str.value
  }
}
// scalastyle:on finalize

/**
 * Helper class to manage multiple executors for data parallelism.
 * @author Yizhi Liu
 * @param symbol output symbol
 * @param ctx devices to run on
 * @param paramNames Name of all trainable parameters of the network.
 * @param argNames Name of all arguments of the network.
 * @param auxNames Name of all auxiliary states of the network.
 * @param trainData Training data iterator.
 * @param workLoadList The list of work load for different devices, in the same order as ctx
 * @param logger When not specified, default logger will be used.
 */
class DataParallelExecutorManager(symbol: Symbol,
                                  ctx: Array[Context],
                                  paramNames: Seq[String],
                                  argNames: Seq[String],
                                  private val auxNames: Seq[String],
                                  trainData: DataIter,
                                  private var workLoadList: Seq[Float] = null,
                                  logger: Logger = DataParallelExecutorManager.logger) {
  // preparation
  private val numDevice = ctx.length
  logger.info(s"Start training with [${ctx.mkString(",")}]")

  // make sure the architecture is valid
  Executor.checkArguments(symbol)

  if (workLoadList == null) {
    workLoadList = Seq.fill(numDevice)(1f)
  }
  require(workLoadList.size == numDevice, "Invalid settings for work load.")

  private val slices = Executor.splitInputSlice(trainData.batchSize, workLoadList)

  private val trainExecs =
    ctx.zipWithIndex.map { case (context, i) =>
      val dataShapes =
        (trainData.provideData ++ trainData.provideLabel).map { case (name: String, shape: Shape) =>
          (name, Shape(slices(i)._2 - slices(i)._1) ++ shape.drop(1))
        }
      symbol.simpleBind(context, "write", shapeDict = dataShapes)
    }

  // data structure
  private val dataNames = trainData.provideData.map(_._1).toArray
  private val labelNames = trainData.provideLabel.map(_._1).toArray

  private val dataArrays =
    dataNames.map { name =>
      trainExecs.zipWithIndex.map { case (exec, i) =>
        val slice = slices(i)
        (slice._1, slice._2, exec.argDict(name))
      }
    }
  private val labelArrays =
    labelNames.map { name =>
      trainExecs.zipWithIndex.map { case (exec, i) =>
        val slice = slices(i)
        (slice._1, slice._2, exec.argDict(name))
      }
    }

  private val paramIdx = (0 until argNames.length).filter { i =>
    paramNames.contains(argNames(i))
  }
  private[mxnet] val _paramNames = paramIdx.map(argNames(_))
  private[mxnet] val paramArrays = paramIdx.map { i =>
    trainExecs.map(_.argArrays(i))
  }.toArray
  private[mxnet] val gradArrays = paramIdx.map { i =>
    trainExecs.map(_.gradArrays(i))
  }.toArray

  private val auxArrays = (0 until auxNames.length).map { i =>
    trainExecs.map(_.auxArrays(i))
  }.toArray
  private val batchSize = trainData.batchSize
  private val outputShapes: Array[Shape] = trainExecs(0).outputs.map { x: NDArray =>
      Shape(batchSize) ++ x.shape.drop(1)
    }
  private[mxnet] val cpuOutputArrays = outputShapes.map(NDArray.zeros(_))

  /**
   * Release the related executors.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    trainExecs.foreach(_.dispose())
  }

  // Install monitor on all executors
  def installMonitor(monitor: Monitor): Unit = {
    trainExecs.foreach(monitor.install)
  }

  /**
   * Set parameter and aux values
   * @param argParams source parameter arrays
   * @param auxParams source aux arrays
   */
  def setParams(argParams: Map[String, NDArray], auxParams: Map[String, NDArray]): Unit = {
    trainExecs.foreach(_.copyParamsFrom(argParams, auxParams))
  }

  /**
   * Copy data from each executor to `arg_params` and `aux_params`
   * @param argParams target parameter arrays
   * @param auxParams target aux arrays
   * @note This function will inplace update the NDArrays in arg_params and aux_params.
   */
  def copyTo(argParams: Map[String, NDArray], auxParams: Map[String, NDArray]): Unit = {
    for ((name, block) <- _paramNames zip paramArrays) {
      val weight = block.map(_.copyTo(Context.cpu())).reduce(_ + _) / block.length
      weight.copyTo(argParams(name))
    }
    for ((name, block) <- auxNames zip auxArrays) {
      val weight = block.map(_.copyTo(Context.cpu())).reduce(_ + _) / block.length
      weight.copyTo(auxParams(name))
    }
  }

  // load data and labels into arrays
  def loadDataBatch(dataBatch: DataBatch): Unit = {
    Executor.loadDataMulti(dataBatch, dataArrays)
    Executor.loadLabelMulti(dataBatch, labelArrays)
  }

  // Perform a forward pass on each executor
  def forward(isTrain: Boolean = false): Unit = {
    for ((texec, islice) <- trainExecs zip slices) {
      texec.forward(isTrain)
      for ((cpuOut, devOut) <- cpuOutputArrays zip texec.outputs) {
        devOut.copyTo(cpuOut.slice(islice))
      }
    }
  }

  // Perform a backward pass on each executor
  def backward(): Unit = {
    trainExecs.foreach(_.backward())
  }
}

object DataParallelExecutorManager {
  private val logger = LoggerFactory.getLogger(classOf[Model])
}
