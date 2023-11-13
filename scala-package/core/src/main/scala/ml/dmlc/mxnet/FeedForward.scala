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
        argPar