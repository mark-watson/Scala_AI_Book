# Ollama Vision: Read Text from Images

The Ollama chapter sends text and gets text. Newer Ollama builds take images too: attach base64 PNGs to a chat call and ask a vision model to read them. This chapter adds that path in Scala 3. Send a photo of a ticket, a scan, or a chart, and read back plain text for the RAG or tools chapters to use.

All code is in `source-code/ollama-vision`.

## Encode, Attach, Ask

Three steps, each its own function so tests can pin each one. `encodeImage` reads bytes and base64 codes them, raising fast on a lost path. `chatPayload` nests the images array in a user message with stream off. `parseReply` reads `message.content` and lifts server side errors instead of swallowing them:

```scala
def chatPayload(model: String, prompt: String, imagesBase64: List[String]): ujson.Value =
  ujson.Obj(
    "model" -> model,
    "stream" -> false,
    "messages" -> ujson.Arr(
      ujson.Obj("role" -> "user", "content" -> prompt,
        "images" -> ujson.Arr.from(imagesBase64.map(ujson.Str(_))))
    )
  )
```

`imageToText` takes one path or a list, so compare calls send two shots at once. The default model and host read `OLLAMA_MODEL` and `OLLAMA_HOST`, which keeps scripts free of local paths.

## Pick a Model That Sees

Text models reject the images field, so pull a vision build first, such as `qwen3-vl:2b` for small GPUs or `llava` where it fits. Ask for plain text out when you plan to feed RAG: "Print out the plain text in this image." Terse prompts beat neat ones here, since chatty framing leaks into your chunks.

Run the demo and the checks:

```bash
cd source-code/ollama-vision
scala-cli run . --main-class ollamavision.visionDemo
VISION_IMAGE=/path/to/pic.png scala-cli run . --main-class ollamavision.visionDemo
scala-cli run . --main-class ollamavision.visionTest
```

The tests code a 1x1 PNG to a temp file and check the base64 round trip, the payload shape, and both error paths, all with no server.
