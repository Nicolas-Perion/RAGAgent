from graph import graph

QUERY = "Who is Sherlock Holmes ?"

for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": QUERY
            }
        ]
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        message = update["messages"][-1]
        if isinstance(message.content, list):
            message.content = [
                {"type": "text", "text": block["text"]} if block.get("type") == "text" else block
                for block in message.content
                if isinstance(block, dict) and block.get("type") in ["text", "tool_use", "tool_result"]
            ]
        message.pretty_print()