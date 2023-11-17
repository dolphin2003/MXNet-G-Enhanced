package ml.dmlc.mxnet

import ml.dmlc.mxnet.io.NDArrayIter
import ml.dmlc.mxnet.optimizer.SGD
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.mutable.ListBuffer

/**
 * Model class of MXNet for training and predicting feedforward nets.
 * This class is designed for a single-data single output supervised network.
 * @param symbol The symbol configuration of computation network.
 * @param ctx The device context of training and prediction.
 *            To use multi GPU training, pass in a list of gpu contexts.
 * @param numEpoch Training parameter, number of training epochs(epochs).
 * @param epochSize Number of batches in a epoch. In default, it is set to
 *                  ceil(num_train_examples / batch_size)
 * @param optimizer Training parameter, name or optimizer object for training.
 * @param initializer Training parameter, the initialization scheme used.
 * @param batchSize The batch size of training data.
 * @param argParams Model parameter, dict of name to NDArray of net's weights.
 * @param auxParams Model parameter, dict of name to NDArray of net's auxiliary states.
 * @param allowExtraParams Whether allow extra parameters that are not needed by symbol
 *                         to be passed by aux_params and arg_params.
 *                         If this is True, no error will be thrown when aux_params and arg_params
 *                         contain extra parameters than needed.
 * @param beginEpoch The beginning training epoch.
 */
class FeedForward(val symbol: Symbol, val ctx: Array[Context] = Array(Context.cpu()),
                  val numEpoch: Int = -1, val epochSize: Int = -1,
                  val optimizer: Optimizer = new SGD(),
                  val initializer: Initializer = new Uniform(0.01f),
                  val batchSize: Int = 128,
                  argParams: Map[String, NDArray] = null,
                  auxParams: Map[String, NDArray] = null,
                  allowExtraParams: Boolean = false,
                  val beginEpoch: Int = 0) {
  val logger: Logger = LoggerFactory.getLogger(classOf[FeedForward])
  // check if symbol contain duplicated names.
  Executor.checkArguments(symbol)

  // rematch parameters to delete useless ones
  private var _argParams =
    if (allowExtraParams) {
      if (argParams != null) {
        val argNames = symbol.listArguments().toSet
        argParams.filter { case (k, v) => argNames.contains(k) }
      } else {
        null
      }
    } else {
      argParams
    }
  private var _auxParams =
    if (allowExtraParams) {
      if (auxParams != null) {
        val auxNames = symbol.listAuxiliaryStates().toSet
        auxParams.filter { case (k, v) => auxNames.contains(k) }
      } else {
        null
      }
    } else {
      auxParams
    }

  def getArgParams: Map[String, NDArray] = _argParams
  def getAuxParams: Map[String, NDArray] = _auxParams

  // internal helper state
  var predExec: Executor = null

  private var monitor: Option[Monitor] = None

  def setMonitor(m: Monitor): Unit = {
    monitor = Option(m)
  }

  def unsetMonitor(): Unit = {
    setMonitor(null)
  }

  // Initialize weight parameters and auxiliary states
  private def initParams(inputShapes: Map[String, Shape], overwrite: Boolean = false)
  : (Seq[String], Seq[String], Seq[String]) = {
    val (argShapes, _, auxShapes) = symbol.inferShape(inputShapes)
    val argNames = symbol.listArguments()
    val inputNames = inputShapes.keys
    val paramNames = argNames.toSet -- inputNames.toSet
    val auxNames = symbol.listAuxiliaryStates()

    val paramNameShapes = (argNames zip argShapes).filter { case (name, _) =>
      paramNames.contains(name)
    }
    val argParams = paramNameShapes.map { case (name, shape) =>
      (name, NDArray.zeros(shape))
    }.toMap
    val auxParams = (auxNames zip auxShapes).map { case (name, shape) =>
      (name, NDArray.zeros(shape))
    }.toMap

    for ((k, v) <- argParams) {
      if (_argParams != null && _argParams.contains(k) && (!overwrite)) {
        argParams(k).set(_argParams(k))
      } else {
        initializer(k, v)
      }
    }

    for ((k, v) <- auxParams) {
      if (_auxParams != null && _auxParams.contains(k) && (!overwrite)) {
        auxParams(k).set(_auxParams(k))
      } else {
        initializer(k, v)
      }
    }

    _argParams = argParams
    _auxParams = auxParams
    (argNames, paramNames.toSeq, auxNames)
  }

  // Initialize the predictor module for running prediction.
  private def initPredictor(inputShapes: Map[String, Shape]): Unit = {
    if (this.predExec == null) {
      val predExec = symbol.simpleBind(ctx(0), gradReq = "null", shapeDict = inputShapes)
      predExec.copyParamsFrom(_argParams, _auxParams)
      Executor.checkArguments(symbol)
      this.predExec = predExec
    }
  }

  // Initialize the iterator given input.
  private def initIter(X: NDArray, y: NDArray, isTrain: Boolean): DataIter = {
    require(y != null || !isTrain, "y must be specified")
    val label = if (y == null) NDArray.zeros(X.shape(0)) else y
    require(label.shape.length == 1, "Label must be 1D")
    require(X.shape(0) == label.shape(0), "The numbers of data points and labels not equal")
    if (isTrain) {
      new NDArrayIter(IndexedSeq(X), IndexedSeq(label), batchSize,
        shuffle = isTrain, lastBatchHandle = "roll_over")
    } else {
      new NDArrayIter(IndexedSeq(X), IndexedSeq(label), batchSize, shuffle = false)
    }
  }

  // Initialize the iterator given eval_data.
  private def initEvalIter(evalData: (NDArray, NDArray)): DataIter = {
    if (evalData == null) {
      null
    } else {
      initIter(evalData._1, evalData._2, isTrain = true)
    }
  }

  /**
   * Run the prediction, always only use one device.
   * @param data eval data
   * @param numBatch the number of batch to run. Go though all batches if set -1
   * @return The predicted value of the output.
   *         Note the network may have multiple outputs, thus it return an array of [[NDArray]]
   */
  def predict(data: DataIter, numBatch: Int = -1): Array[NDArray] = {
    data.reset()
    val dataShapes = data.provideData
    val dataNames = dataShapes.map(_._1).toArray
    initPredictor(dataShapes)
    val batchSize = data.batchSize
    val dataArrays = dataNames.map(predExec.argDict(_))
    val outputs = Array.fill(predExec.outputs.length)(ListBuffer.empty[NDArray])

    var i = 0
    while (data.hasNext && i != numBatch) {
      val batch = data.next()
      i += 1
      Executor.loadData(batch, dataArrays)
      predExec.forward(isTrain = false)
      val padded = batch.pad
      val realSize = batchSize - padded
      for ((list, nd) <- outputs zip predExec.outputs) {
        list += nd.slice(0, realSize).copy()
      }
    }
    // TODO(Yizhi): we can use Symbol.concat to do the same thing. Can it be more efficient?
    val results = outputs.map(NDArray.concatenate(_))
    for (output <- outputs) {
      output.foreach(_.dispose())
    }
    results
  }

  /**
   * Fit the model.
   * @param trainData Training data
   * @param evalData Evaluation data
   * @param evalMetric The evaluation metric, cannot be null
   * @param epochEndCallback A callback that is invoked at end of each epoch.
   *                         This can be used to checkpoint model each epoch.
   * @param batchEndCallback A callback that is invoked at end of each batch
   *                         For print purpose
   * @param kvStoreType A string kvstore type:
   *                    'local' : multi-devices on a single machine, will automatically
   *                    choose one from 'local_update_cpu', 'local_allreduce_cpu', and
   *                    'local_allreduce_device'
   *                    'dist_sync' : multi-machines with BSP
   *                    'dist_async' : multi-machines with partical asynchronous
   *                    In default uses 'local', often no need to change for single machine.
   * @param logger When not specified, default logger will be used.
   * @param workLoadList The list of work load for different devices, in the same order as ctx
   */
  def fit(trainData: DataIter, evalData: DataIter, evalMetric: EvalMetric, kvStoreType: String,
          epochEndCallback: EpochEndCallback, batchEndCallback: BatchEndCallback,
          logger: Logger, workLoadList: Seq[Float]): Unit = {
    // create kvstore
    val (kvStore, updateOnKVStore) = Model.createKVStore(kvStoreType, ctx.length, _argParams)
    fit(trainData, evalData, evalMetric, kvStore, updateOnKVStore,
      epochEndCallback, batchEndCallback, logger, workLoadList)
    kvStore.foreach(_.dispose())
  }

  def fit(trainData: DataIter, evalData: DataIter, evalMetric: EvalMetric,
          kvStoreType: String, epochEndCallback: EpochEndCallback,
          batchEndCallback: BatchEndCallback): Unit = {
    fit(trainData, evalData, evalMetric, kvStoreType,
      epochEndCallback, batchEndCallback, FeedForward.logger, null)
  }

  def fit(trainData: DataIter, evalData: DataIter,
          evalMetric: EvalMetric, kvStoreType: String): Unit = {
    fit(trainData, evalData, evalMetric, kvStoreType,
      epochEndCallback = null, batchEndCallback = null)
  }

  def fit(trainData: DataIter, evalData: DataIter, evalMetric: EvalMetric): Unit = {
    fit(trainData, evalData, evalMetric, kvStoreType = "local")
  }

  def fit(trainData: DataIter, evalData: DataIter): Unit = {
    fit(trainData, evalData, new Accuracy())
  }

  def fit(trainData: DataIter, evalData: DataIter, evalMetric: EvalMetric,
          kv: KVStore,
          epochEndCallback: EpochEndCallback,
          batchEndCallback: BatchEndCallback, logger: Logger,
          workLoadList: Seq[Float]): Unit = {
    // create kvstore
    val (kvStore, updateOnKVStore) = Model.createKVStore(kv)
    fit(trainData, evalData, evalMetric, kvStore, updateOnKVStore,
      epochEndCallback, batchEndCallback, logger, workLoadList)
  }

  def fit(trainData: DataIter, evalData: DataIter, evalMetric: EvalMetric,
          kvStore: KVStore,
          epochEndCallback: EpochEndCallback,
          batchEndCallback: BatchEndCallback): Unit = {
    fit(trainData, evalData, evalMetric, kvStore, epochEndCallback,
        batchEndCallback, FeedForward.logger, null)
  }

  def fit(trainData: DataIter, evalData: DataIter,
          evalMetric: EvalMetric, kvStore: KVStore): Unit = {
    fit(trainData, evalData, evalMetric, kvStore, epochEndCallback = null, batchEndCallback = null)
  }

  def fit(trainData: DataIter, evalData: DataIter, kvStore: KVStore): Unit = {
    fit(trainData, evalData, new Accuracy(), kvStore)
  }