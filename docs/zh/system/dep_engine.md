Execution Engine
================

MXNet 的执行引擎不仅仅是为了深度学习和其他任何特定的领域问题. 相反地, 它设计用来解决通用问题: 根据依赖关系来执行一系列的功能操作. 有依赖关系的任意两个功能需要被序列化. 没有依赖的功能 *可以* 并发执行来提升系统性能. 也可以参考 [Note on Dependency Engine](note_engine.md).

Interface
---------
执行引擎的核心接口如下:

```c++
virtual void PushSync(Fn exec_fun, Context exec_ctx,
                      std::vector<VarHandle> const& const_vars