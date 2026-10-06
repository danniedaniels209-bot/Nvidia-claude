"""Layerize: turn a flat (e.g. AI-generated) image into a library of ready-to-use layers.

Heavy models run on a Colab GPU (`server.py`); the local machine only runs the tiny
MCP client (`mcp_server.py`) that fetches the layers an agent asks for.
"""
