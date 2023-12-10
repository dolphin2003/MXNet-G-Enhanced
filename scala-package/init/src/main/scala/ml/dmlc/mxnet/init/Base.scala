package ml.dmlc.mxnet.init

object Base {
  tryLoadInitLibrary()
  val _LIB = new LibInfo

  // type definitions
  class RefInt(val value: Int = 0)
  class RefLong(val value: Long = 0)
  class RefFloat(val value: Float = 0)
  class RefString(val value: String = null)

  type CPtrAddress = Long

  type NDArrayHandle = CPtrAddress
  type FunctionHandle = CPtrAddress
  type KVStoreHandle = CPtrAddress
  type ExecutorHandle = CPtrAddress
  type SymbolHandle = CPtrAddress

  @throws(classOf[UnsatisfiedLinkError])
  private def tryLoadInitLibrary(): Unit