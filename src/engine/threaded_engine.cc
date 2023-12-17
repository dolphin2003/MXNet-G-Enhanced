/*!
 * Copyright (c) 2015 by Contributors
 * \file threaded_engine.cc
 * \brief implements base threaded engine.
 * \author Yutian Li
 */
#include <dmlc/logging.h>
#include <cassert>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <utility>
#include "./threaded_engine.h"
#include "../common/cuda_utils.h"

namespace mxnet {
namespace engine {

#if ENGINE_DEBUG
std::atomic<std::size_t> OprBlock::counter{0};
std::atomic<std::size_t> VersionedVarBlock::counter{0};
std::atomic<std::size_t> ThreadedVar::counter{0};
std::atomic<std::size_t> ThreadedOpr::counter{0};
#endif  // ENGINE_DEBUG

ThreadedVar::ThreadedVar(VersionedVarBlock* head) : head_{head} {
#if ENGINE_DEBUG
  LOG(INFO) << __func__ << " " << ++counter;
#endif  // ENGINE_DEBUG
}

inline void ThreadedVar::AppendReadDependency(OprBlock* opr_block) {
  std::lock_guard<std::mutex> lock{m_};
  if (pending_write_ == nullptr) {
    // invariant: is_ready_to_read()
    CHECK_GE(num_pending_reads_, 0);
    // STATE CHANGE
    ++num_pending_reads_;
    // decrease wait counter
    opr_block->decr_wait();
  } else {
    auto&& new_var_block = VersionedVarBlock::New();
    assert(head_->next == nullptr);
    assert(head_->trigger == nullptr);
    assert(head_->write == false);
    // append things to next.
    head_->next = new_var_block;
    head_->trigger = opr_block;
    head_ = new_var_block;
  }
}

inline void ThreadedVar::AppendWriteDependency(OprBlock* opr_block) {
  auto&& new_var_block = VersionedVarBlock::New();
  std::lock_guard<std::mutex> lock{m_};
  // invariant.
  assert(head_->next == nullptr);
  assert(head_->trigger == nullptr);
  assert(head_->write == false);
  // attach to head.
  head_->next = new_var_block;
  head_->trigger = opr_block;
  head_->write = true;

  // check if it is ready to write
  if (pending_write_ == nullptr) {
    // invariant: is_ready_to_read()
    pending_write_ = head_;
    CHECK_GE(num_pending_reads_, 0);
    if (num_pending_reads_ == 0) {
      // STATE CHANGE
      opr_block->decr_wait();
      num_pending_reads_ = kWriteTriggered;
    }
  } else {
    CHECK_NE(num_pending_reads_, 0);
  }
  head_ = new_var_block;
}

template <typename Dispatcher>
inline void ThreadedVar::CompleteReadDependency(Dispatcher dispatcher) {
  OprBlock *trigger = nullptr;
  {
    // this is lock scope
    std::lock_guard<std::mutex> lock{m_};
    CHECK_GT(num_pending_reads_, 0);

    if (--num_pending_reads_ == 0) {
      if (pending_write_ != nullptr) {
        // STATE CHANGE
        trigger = pending_write_->trigger;
        num_pending_reads_ = kWriteTriggered;
      }
    }
  }
  if (trigger != nullptr && trigger->decr_wait() == 0) {
    dispatcher(trigger);
  }
}

template <typename Dispatcher>
inline bool ThreadedVar::CompleteWriteDependency(Dispatcher dispatcher) {
  // this is lock scope
  VersionedVarBlock *old_pending_write, *end_of_read_chain;
  OprBlock* trigger_write = nullptr;
  {
    std::lock_guard<std::mutex> lock{m_};
    // invariants
    assert(head_->next == nullptr);
    assert(pending_write_ != nullptr);
    CHECK_EQ(num_pending_reads_, kWriteTriggered);

    // really delete
    if (to_delete_) {
      VersionedVarBlock *head = pending_write_->next;
      VersionedVarBlock::Delete(pending_write_);
      assert(head_ == head);
      VersionedVarBlock::Delete(head);
      return true;
    }
    // detach pending write
    old_pending_write = pending_write_;
    // search for chains to trigger
    end_of_read_chain = old_pending_write->next;
    // reset to 0 pending reads
    num_pending_reads_ = 0;
    while (end_of_read_chain != head_ &&
           end_of_read_chain->write == false) {
      ++num_pending_reads_;
      end_of_read_chain = end_of_read_chain->next;
    }
    if (end_of_read_chain == head_) {
      pending_write_ = nullptr;
    } else {
      // check if there is pending reads, if not trigger write
      assert(end_of_read_chain->write == true);
      pending_write_ = end_of_read_chain;
      if (num_pending_reads_ == 0) {
        // mark write as already actived in this var
        num_pending_reads_ = kWriteTriggered;
        trigger_write = end_of_read_chain->trigger;
      }
    }
  }
  // This is outside of lock scope
  // Be very carful, pending_write_ and num_pending_reads_
  // can change now, do not reply ont the two variables.
  // The linked list \in [old_pending_write, end_of_read_chain)
  // is already detached from this Var.
  // So it is safe to modify these
  VersionedVarBlock *cur_head = old_pending_write->next;
  VersionedVarBlock::Delete(old_pending_write);
  // dispatch all the events
  while (cur_head != end_of_read_chain) {
    if (cur_head->trigger->decr_wait() == 0) {
      dispatcher(cur_head->trigger);
    }
    auto prev = cur_head;
    cur_head = cur_head->next;
    assert(cur_head != nullptr);
    VersionedVarBlock::Delete(prev);
  }
  if (trigger_write != nullptr && trigger_write->decr_wait() == 0) {
    dispatcher(trigger_write);
  }
  return false;
}

inline void ThreadedVar::SetToDelete() {
  std::lock_guard<std::mutex> lock{m_};
  to_delete_ = true;
}

inline bool ThreadedVar::ready_to_read() {
  std::lock_guard<std::mutex> lock{m_};
  return this->is_ready_to_read();
}

// implementation of threaded engine
ThreadedVar* ThreadedEngine::NewVariable() {
  return ThreadedVar::New(VersionedVarBlock::New());
}

ThreadedOpr* ThreadedEngine::NewOperator(
    ThreadedEngine::AsyncFn fn,
    std::vector<VarHandle> const& const_vars,
    std::vector<VarHandle> const& mutable_vars,
    FnProperty prop) {
  auto ret = ThreadedOpr::New();
  ret->fn = fn;
  ret->prop = prop;
  ret->const_vars.resize(const_vars.size());
  ret->mutable_vars.resize(mutable_vars.size());
  std::transform(const_vars.begin(), const_vars.end(),
                 ret->const_vars.begin(), ThreadedVar::CastFromBase);
//  std::cout << "---------------begin----------------" << std::endl;
//  for(auto var : ret->const_vars) {
//	  std::cout << var->ready_to_read() << std::endl;
// 