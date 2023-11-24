package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import org.slf4j.{LoggerFactory, Logger}

/**
 * Key value store interface of MXNet for parameter synchronization.
 * @author Yizhi Liu
 */
object KVStore {

  // group id of scheduler/server/worker
  val GROUP_NODE_SCHEDULER = 1
  val GROUP_NODE_SERVER = 2
  val GROUP_NODE_WORKER = 4

  /**
   * Create a new KVStore. <br />
   * <b>
   * WARNING: it is your responsibility to clear this object through dispose().
   * NEVER rely on the GC strategy
   * </b>
   *
   * @param name : {'local', 'dist'}
   *     The type of KVStore
   *     - local works for multiple devices on a single machine (single process)
   *     - dist works for multi-machines (multiple processes)
   * @return The created KVStore
   */
  def create(name: String = "local"): KVStore = {
    val handle = new KVStoreHandleRef
    checkCall(_LIB.mxKVStoreCreate(name, handle))
    new KVStore(handle.value)
  }
}

// scalastyle:off finalize
class KVStore(private[mxnet] val handle: KVStoreHandle) {
  private val logger: Logger = LoggerFactory.getLogger(classOf[KVStore])
  private var updaterFunc: MXKVStoreUpdater = null
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
      _LIB.mxKVStoreFree(handle)
      disposed = true
    }
  }

  /**
   * Initialize a single or a sequence of key-value pairs into the store.
   * For each key, one must init it before push and pull.
   * Only worker 0's (rank == 0) data are used.
   * This function returns after data have been initialized successfully
   *
   * @param keys The keys.
   * @param values The values.
   */
  def init(keys: Array[Int], values: Array[NDArray]): Unit = {
    require(keys.length == values.length, "len(keys) != len(values)")
    val valuePtrs = values.map(_.handle)
    checkCall(_LIB.mxKVStoreInit(handle, keys.length, keys, valuePtrs))
  }

  def init(key: Int, value: NDArray): Unit = {
    init(Array(key), Array(value))
  }

  /**
   * Push a single or a sequence of key-value pairs into the store.
   * Data consistency:
   * 1. this function returns after adding an operator to the engine.
   * 2. push is always called after all previous push and pull on the same key are finished
   * 3. there is no synchronization between workers. One can use _barrier() to sync all workers
   *
   * @param keys Keys
   * @param values  According 