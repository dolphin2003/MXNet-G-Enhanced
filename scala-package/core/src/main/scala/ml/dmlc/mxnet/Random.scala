package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import ml.dmlc.mxnet.NDArray.{randomGaussian, randomUniform, empty}

/**
 * Random Number interface of mxnet.
 * @author Yuan Tang
 */
object Random {
  /**
   * Generate uniform distribution in [low, high) with shape.
   *
   * @param low The lower bound of distribution.
   * @param high The upper bound of distribution.
   * @param shape Output shape of the NDArray generated.
   * @param ctx Context of output NDArray, will use default context if not specified.
   * @param out Output place holder
   * @return The result NDArray with generated result.
   */
  def uniform(low: Float,
              high: Float,
              shape: Shape = null,
              ctx: Context = null,
              out: NDArray = null): NDArray = {
    var outCopy = out
    if (o