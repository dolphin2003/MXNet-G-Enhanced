package ml.dmlc.mxnet.spark.io

import ml.dmlc.mxnet.{NDArray, DataBatch, DataIter, Shape}
import org.apache.spark.mllib.linalg.Vector

import scala.collection.mutable.ArrayBuffer

/**
 * A temporary helper implementation for predicting Vectors
 * @author Yizhi Liu
 */
class PointIter private[mxnet](
  private val points: Iterator[Vector],
  private val dimension: Shape,
  private val _batchSize: Int,
  private val dataName: String = "data",
  private val labelName: String = "label") extends DataIter {

  private val cache: ArrayBuffer[DataBatch] = ArrayBuffer.empty[DataBatch]
  private var index: Int = -1
  private val dataShape = Shape(_batchSize) ++ dimension

  def dispose(): Unit = {
    cache.foreach(_.dispose())
  }

  /**
   * reset the iterator
   */
  override def reset(): Unit = {
    index = -1
  }

  @throws(classOf[NoSuchElementException])
  override def next(): DataBatch = {
    if (!hasNext) {
      throw new NoSuchElementExcept