FROM python:3.11-slim

WORKDIR /app
COPY . .

# Install with all optional solvers
RUN pip install --no-cache-dir -e ".[full]"

EXPOSE 8080

# Start MCP server on HTTP/SSE transport
CMD ["math-logic-mcp", "--http"]
