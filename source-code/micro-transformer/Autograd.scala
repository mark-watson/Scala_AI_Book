//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package microtransformer

import scala.collection.mutable

// Scalar autograd engine. Port of the Value struct and v+/v*/vpow/
// vlog/vexp/vrelu/backward code in microgpt.lisp.
class Value(var data: Double, var grad: Double = 0.0):
  private var children: Set[Value] = Set.empty
  private var backwardFn: () => Unit = () => ()

  private def result(d: Double, kids: Set[Value], back: Value => () => Unit): Value =
    val v = new Value(d)
    v.children = kids
    v.backwardFn = back(v)
    v

  def +(other: Value): Value =
    val a = this; val b = other
    result(data + b.data, Set(a, b), out => () => { a.grad += out.grad; b.grad += out.grad })
  def *(other: Value): Value =
    val a = this; val b = other
    result(data * b.data, Set(a, b), out => () => { a.grad += b.data * out.grad; b.grad += a.data * out.grad })
  def pow(p: Double): Value =
    val a = this
    result(math.pow(data, p), Set(a), out => () => a.grad += p * math.pow(a.data, p - 1) * out.grad)
  def exp(): Value =
    val a = this
    val e = math.exp(data)
    result(e, Set(a), out => () => a.grad += e * out.grad)
  def log(): Value =
    val a = this
    result(math.log(data), Set(a), out => () => a.grad += out.grad / a.data)
  def relu(): Value =
    val a = this
    result(math.max(0.0, data), Set(a), out => () => a.grad += (if a.data > 0.0 then out.grad else 0.0))
  def unary_- : Value = this * Value(-1.0)
  def -(other: Value): Value = this + (-other)
  def /(other: Value): Value = this * other.pow(-1.0)

  def backward(): Unit =
    val topo = mutable.ListBuffer[Value]()
    val seen = mutable.Set[Value]()
    def build(v: Value): Unit =
      if !seen.contains(v) then
        seen.add(v)
        v.children.foreach(build)
        topo.append(v)
    build(this)
    grad = 1.0
    topo.reverse.foreach(_.backwardFn())

object Value:
  def apply(d: Double): Value = new Value(d)

  // Softmax over one row of Values. Each output holds the graph.
  def softmax(logits: Seq[Value]): Seq[Value] =
    val m = logits.map(_.data).max
    val exps = logits.map(v => (v - Value(m)).exp())
    val s = exps.reduce(_ + _)
    exps.map(_ / s)
