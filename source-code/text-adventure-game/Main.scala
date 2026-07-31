// Copyright 2025-2026 Mark Watson. All rights reserved.
//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::upickle:4.4.3

package textadventure

import upickle.default.*
import ujson.*

/** Represents a chat message in the conversation history */
enum Role(val value: String):
  case System extends Role("system")
  case User extends Role("user")
  case Assistant extends Role("assistant")

case class ChatMessage(role: Role, content: String)

object ChatMessage:
  given ReadWriter[ChatMessage] = readwriter[ujson.Value].bimap[ChatMessage](
    msg => Obj("role" -> msg.role.value, "content" -> msg.content),
    json =>
      val obj = json.obj
      val roleStr = obj("role").str
      val content = obj("content").str
      val role = Role.values
        .find(_.value == roleStr)
        .getOrElse(throw RuntimeException(s"Unknown role: $roleStr"))
      ChatMessage(role, content)
  )

/** Ollama chat client that handles conversation history */
object OllamaChatClient:
  private val DEFAULT_MODEL = "qwen3.5:4b"
  private val DEFAULT_BASE_URL = "http://localhost:11434"

  /** Send a chat completion request to Ollama with the given message history */
  def getChat(
      messages: List[ChatMessage],
      model: String = DEFAULT_MODEL,
      baseUrl: String = DEFAULT_BASE_URL
  ): Either[String, String] =

    val url = s"$baseUrl/api/chat"

    // Build the payload with all messages as ujson.Arr
    val messagesJson = Arr(messages.map(msg => writeJs(msg))*)
    val payload = Obj(
      "model" -> model,
      "messages" -> messagesJson,
      "stream" -> false
    ).render()

    try
      val response = requests.post(
        url = url,
        headers = Map("Content-Type" -> "application/json"),
        data = payload,
        readTimeout = 300000 // 5 minutes timeout for longer responses
      )

      if response.statusCode == 200 then
        val json = ujson.read(response.text())

        // Handle streaming vs non-streaming response
        val content = json("message")("content").str
        Right(content)
      else
        Left(
          s"Ollama API request failed (HTTP ${response.statusCode}): ${response.text()}"
        )
    catch
      case e: Exception =>
        Left(
          s"Error connecting to Ollama: ${e.getMessage}. Please ensure Ollama is running."
        )

object TextAdventureGame:

  val STORY_PROMPT_FILE = "story.txt"
  val DEFAULT_MODEL = sys.env.getOrElse("TEXT_ADV_OLLAMA_MODEL", "qwen3.5:4b")

  /** Load the system prompt from story.txt */
  def loadStoryPrompt(): Either[String, String] =
    try
      val content = scala.io.Source.fromFile(STORY_PROMPT_FILE).mkString
      Right(content)
    catch
      case _: java.io.FileNotFoundException =>
        Left(
          s"Could not find $STORY_PROMPT_FILE. Please ensure it exists in the project directory."
        )
      case e: Exception =>
        Left(s"Error reading story prompt: ${e.getMessage}")

  /** Run the interactive text adventure game */
  @main def play(): Unit =
    println("=" * 60)
    println("TEXT ADVENTURE GAME")
    println("=" * 60)

    // Load system prompt
    val prompt = loadStoryPrompt() match
      case Left(err) =>
        println(s"Error: $err")
        return
      case Right(p) => p

    val model = DEFAULT_MODEL
    println(s"\nUsing Ollama model: $model")
    println("Ensure Ollama is running (ollama serve)")

    // Initialize conversation with system prompt as the first message
    var history: List[ChatMessage] =
      List(ChatMessage(Role.System, prompt))

    println("\n" + "-" * 60)
    println("Welcome to the Text Adventure!")
    println("Type your actions at the > prompt.")
    println("Type 'quit', 'exit', or 'q' to end the game.")
    println("-" * 60)

    // Get initial response from game master (starts with just system message)
    print("\n> ")

    var running = true
    while running do
      val input = scala.io.StdIn.readLine()

      if input == null || input.trim.isEmpty then
        // Empty line, continue
        print("> ")
      else
        val trimmed = input.trim

        if trimmed.toLowerCase == "quit" || trimmed.toLowerCase == "exit" || trimmed.toLowerCase == "q"
        then
          println("\nThanks for playing!")
          running = false
        else
          history = history :+ ChatMessage(Role.User, trimmed)

          // Get response from Ollama
          print("\nProcessing...\n")

          OllamaChatClient.getChat(history, model) match
            case Left(error) =>
              println(s"Error: $error")
              history =
                history.dropRight(1) // Remove the player message if failed

            case Right(response) =>
              // Display response (clean up any trailing whitespace)
              val cleanResponse = response.trim
              println("\n" + cleanResponse)

              // Add assistant response to history
              history = history :+ ChatMessage(Role.Assistant, cleanResponse)

          print("> ")

  /** Test the connection to Ollama */
  @main def testOllama(model: String = "qwen3.5:4b"): Unit =
    println(s"Testing Ollama connection with model '$model'...")

    val messages = List(
      ChatMessage(Role.System, "You are a helpful assistant."),
      ChatMessage(Role.User, "Hello! Please respond with a brief greeting.")
    )

    OllamaChatClient.getChat(messages, model) match
      case Right(response) =>
        println(s"Success!\nResponse:\n$response")
      case Left(error) =>
        println(s"Failed: $error")
