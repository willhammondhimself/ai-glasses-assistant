"""HUD WebSocket handler for streaming agent outputs to AR glasses."""
from fastapi import WebSocket
from .manager import ConnectionManager, WebSocketHandler, ws_manager
import logging

logger = logging.getLogger(__name__)


class HUDHandler(WebSocketHandler):
    """
    Stream agent outputs to HUD clients (glasses or browser demo).

    Supports:
    - Mode subscriptions (poker, homework, code, all)
    - Direct agent requests via WebSocket
    - Broadcast updates from agent_chat endpoint
    """

    async def handle_message(self, websocket, topic, data, conn_info):
        """Handle incoming HUD client messages."""
        msg_type = data.get("type")

        if msg_type == "subscribe_mode":
            # Subscribe to specific mode updates (poker, homework, code, all)
            mode = data.get("mode", "all")
            conn_info.metadata["mode"] = mode
            await websocket.send_json({
                "type": "subscribed",
                "mode": mode,
                "message": f"Subscribed to {mode} updates"
            })
            logger.info(f"HUD client subscribed to mode: {mode}")

        elif msg_type == "pong":
            # Heartbeat response - already handled by base class
            pass

        elif msg_type == "agent_request":
            # Forward to agent and stream result back
            message = data.get("message", "")
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "message": "Message required"
                })
                return

            try:
                from backend.agent import get_agent
                agent = get_agent()
                result = await agent.run(message)

                await websocket.send_json({
                    "type": "agent_response",
                    "data": result
                })

                # Also broadcast to all HUD clients
                await self.broadcast_hud_update(result)

            except Exception as e:
                logger.error(f"Agent request failed: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })

        elif msg_type == "ping":
            await websocket.send_json({"type": "pong"})

        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            })

    async def broadcast_hud_update(self, data: dict, mode: str = "all"):
        """
        Broadcast update to all HUD clients.

        Args:
            data: The data to broadcast
            mode: Filter to clients subscribed to this mode (or "all")
        """
        await ws_manager.broadcast("hud:global", {
            "type": "hud_update",
            "mode": mode,
            "data": data
        })


# Singleton handler
hud_handler = HUDHandler()


async def broadcast_to_hud(data: dict, mode: str = "all"):
    """
    Helper function to broadcast to HUD from other modules.

    Usage:
        from backend.websocket.hud_handler import broadcast_to_hud
        await broadcast_to_hud(result, mode="poker")
    """
    await hud_handler.broadcast_hud_update(data, mode)
