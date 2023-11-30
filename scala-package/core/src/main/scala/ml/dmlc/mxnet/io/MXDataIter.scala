package ml.dmlc.mxnet.io

import ml.dmlc.mxnet.Base._
import ml.dmlc.mxnet.{DataPack, DataBatch, DataIter, NDArray, Shape}
import ml.dmlc.mxnet.IO._
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer

/**
 * DataIter built in MXNet.
 * @param handle the handle to the underlying C++ Data Iterator
 */
// scalastyle:off finalize
class MXDataIter private[mxnet](private[mxnet] val handle: DataIterHandle,
                                private val dataName: String = "data",
                                private val labelName: String = "label") extends DataIter {
  private val logger = LoggerFactory.getLogger(classOf[MXDataIter])

  // use currentBatch to implement hasNext
  // (may be this is not the best way to do this work,
  // fix me if any better way found)
  private var currentBatch: DataBatch = null

  private val (_provideData: Map[String, Shape],
               _provideLabel: Map[String, Shape],
               _batchSize: Int) =
    if (hasNext) {
      iterNext()
      val data = currentBatch.data(0)
      val label = currentBatch.label(0)
      // properties
      val res = (Map(dataName -> data.shape), Map(labelName -> label.shape), data.shape(0))
      currentBatch.dispose()
      reset()
      res
    } else {
      (null, null, 0)
    }

  private var disposed = false
  override protected def finalize(): Unit = {
    dispose()
  }

  /**
   * Release the native memory.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    if (!disposed) {
      _LIB.mxDataIterFree(handle)
      disposed = true
    }
  }

  /**
   * reset the iterator
   */
  override def reset(): Unit = {
    currentBatch = null
    checkCall(_LIB.mxDataIterBeforeFirst(handle))
  }

  @throws(classOf[NoSuchElementException])
  override def next(): DataBatch = {
    if (currentBatch == null) {
      iterNext()
    }

    if (currentBatch != null) {
      val batch = currentBatch
      currentBatch = null
      batch
    } else {
      throw new NoSuchElementException
    }
  }

  /**
   * Iterate to next batch
   * @return whether the move is successful
   */
  private def iterNext(): Boolean = {
    val next = new RefInt
    checkCall(_LIB.mxDataIterNext(handle, next))
    currentBatch = null
    if (next.value > 0) {
      currentBatch = new DataBatch(data = getData(), label = getLabel(),
        index = getIndex(), pad = getPad())
    }
    next.value > 0
  }

  /**
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = {
    val out = new NDArrayHandleRef
    checkCall(_LIB.mxDataIterGetData(handle, out))
    IndexedSeq(new NDArray(out.value, writable = false))
  }

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = {
    val out = new NDArrayHandleRef
    checkCall(_LIB.mxDataIterGetLabel(handle, out))
    IndexedSeq(new NDArray(out.value, writable = false))
  }

  /**
   * Get the index of current batch
   * @return the index of current batch
   */
  overr