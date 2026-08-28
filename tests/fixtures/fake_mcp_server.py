import json
import sys


def send(envelope):
    body = json.dumps(envelope).encode()
    sys.stdout.buffer.write(b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n")
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def read_envelope():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n", b""):
            break
        key, _, value = line.decode().partition(":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length == 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body)


def main():
    while True:
        env = read_envelope()
        if env is None:
            return
        method = env.get("method")
        req_id = env.get("id")
        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"serverInfo": {"name": "fake", "version": "0.0.1"}},
                },
            )
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "Echo args",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"x": {"type": "string"}},
                                },
                            }
                        ]
                    },
                },
            )
        elif method == "tools/call":
            args = env["params"]["arguments"]
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(args)}],
                        "isError": False,
                    },
                },
            )
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": "unknown method " + str(method)},
                },
            )


if __name__ == "__main__":
    main()
