package ml.dmlc.mxnet.optimizer

import ml.dmlc.mxnet.{NDArray, Optimizer, LRScheduler}
import ml.dmlc.mxnet.NDArrayConversions._

/**
 * Adam optimizer as described in [King2014]
 *
 * [King2014] Diederik Kingma, Jimmy Ba,
 * Adam: A Method for Stochastic Optimization,
 * http://arxiv.org/abs/1412.6980
 *
 * @author Yuan Tang, Yizhi Liu
 *
 * @param learningRate Float, Step size.
 * @param beta1 Float, Exponential decay rate for the first moment estimates.
 * @param beta2 Float, Exponential decay rate for the second moment estimates.
 * @param epsilon Float
 * @param decayFactor Float
 * @param wd Float, L2 regularization coefficient add to all the weights
 * @param clipGradient Float, clip gradient in range [-clip_gradient, clip_gradient]
 * @param lrScheduler The learning rate scheduler
 */
class Adam(val learningRate: Float = 0.002f, val beta1: Float = 0.9f, val beta2: Float = 0.999f,
           val epsilon: Float = 1e-8f, val decayFactor: Float = 1-1e-8f, val wd: Float = 0.0f,
           val clipGradient: Float = 0f, val lrScheduler: LRScheduler = null) extends Optimizer {

  protected var time: Int = 0
  protected var timeFirstIndex: Option[Int] = None

  if (lrScheduler != null) {
    lrScheduler.baseLR = learningRate
  }

  /**
   * Update the parameters.
   * @param index An unique integer key used to index the parameters
   * @param weight weight ndarray
   * @param grad grad ndarray
   * @param state NDArray or other objects returned by initState
   *              The auxiliary state used in optimization.
   */
  override def update(index: Int, weight: NDArray, grad: NDArray, state: AnyRef): Unit = {
    val lr =
      (if (lrScheduler != null) {
        val scheduledLr = lrScheduler(numUpdate)
        updateCount(index)
        scheduledLr
      } else {
        this.learningRate
      }) * lrScale.getOrElse(index, 1f)

    val (mean, variance) = state.asInstanceOf[(NDArray, NDArray)]

    // increment time only when the first parameters is called
    timeFirstIndex match {
      case Some(idx) =>
        if (idx == index) time += 1
      case None =>
        timeFirstIndex = Option(index)
        time = 0 // all parameters share the same time
    }

    val t1: Int = time + 1
    val learningRate = (lr *
      math.sqrt(1.0 - math.pow(beta2, t1)) /
      (1.0 - math.pow(beta1, t1))).toFloat
    val beta1t = beta1 * math.pow(decayFactor, t1 - 1).toFloat

    var res