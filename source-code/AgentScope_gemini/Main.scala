// Copyright 2025-2026 Mark Watson. All rights reserved.

package agentscope_gemini

import io.agentscope.core.ReActAgent
import io.agentscope.core.message.Msg

@main def agentScopeDemo(): Unit =
  val model = GeminiConfig.createModel()

  val agent = ReActAgent.builder()
    .name("Assistant")
    .sysPrompt("You are a helpful AI assistant.")
    .model(model)
    .build()

  val response = agent.call(
    Msg.builder()
      .textContent("Hello! Tell me a fun fact about Scala programming.")
      .build()
  ).block()

  println("=== Agent Response ===")
  println(response.getTextContent())
