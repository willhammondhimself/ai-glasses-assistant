"""Poker stream WebSocket handler for real-time webcam OCR and HUD updates."""

import json
import asyncio
import logging
from typing import Dict, Set, Optional, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

from backend.vision.poker_ocr import poker_ocr, poker_router, ev_calculator, spr_calculator, PokerTableState
from backend.agent.tools.poker import PokerTool
from backend.poker.villain_tracker import villain_tracker
from backend.poker.hand_history import hand_history_parser
from backend.poker.table_manager import table_manager
from backend.poker.live_mode import live_mode
from backend.voice.tools.news_tool import PerplexityNewsTool

logger = logging.getLogger(__name__)

# Initialize news tool for ticker
_news_tool = None
def get_news_tool():
    global _news_tool
    if _news_tool is None:
        _news_tool = PerplexityNewsTool()
    return _news_tool


class PokerStreamHandler:
    """Handles WebSocket connections for real-time poker HUD streaming.

    Protocol:
    Client sends:
    - {"type": "start_stream"} - Start 5fps webcam OCR
    - {"type": "stop_stream"} - Stop streaming
    - {"type": "frame", "image": "<base64>"} - Send webcam frame
    - {"type": "manual_input", "hole_cards": "AhKd", "board": "QsJhTc", "pot": 100, "bet": 25}
    - {"type": "vpip_action", "action": "fold|call|raise", "is_raise": false}
    - {"type": "get_vpip_stats"}
    - {"type": "reset_session"} - Reset session VPIP/stats
    - {"type": "get_session_summary"} - Get full session summary
    - {"type": "villain_action", "seat": "Seat 1", "street": "preflop", "action": "raise", ...}
    - {"type": "get_villain_stats"} - Get all villain profiles
    - {"type": "get_news", "count": 5} - Get news headlines for ticker
    - {"type": "set_poker_site", "site": "ignition|generic"} - Switch OCR mode

    Server sends:
    - {"type": "connected", "message": "Poker HUD ready"}
    - {"type": "table_state", "data": {...}} - OCR results
    - {"type": "ev_table", "data": {...}} - EV calculations
    - {"type": "vpip_stats", "data": {...}} - VPIP/PFR leak analysis
    - {"type": "hero_stats", "data": {...}} - Real-time hero session stats
    - {"type": "session_reset", "data": {...}} - Session reset confirmation
    - {"type": "villain_stats", "seat": "...", "data": {...}} - Single villain update
    - {"type": "villain_bulk", "profiles": {...}} - All villain profiles
    - {"type": "news_headlines", "headlines": [...]} - News for ticker
    - {"type": "table_dynamics", "data": {...}} - Table analysis
    - {"type": "error", "message": "..."}
    """

    def __init__(self):
        self.connections: Set[WebSocket] = set()
        self.active_streams: Dict[WebSocket, bool] = {}
        self.poker_tool = PokerTool()
        self._frame_interval = 0.2  # 5fps = 200ms between frames

    async def handle_connection(self, websocket: WebSocket):
        """Handle a new WebSocket connection."""
        await websocket.accept()
        self.connections.add(websocket)
        self.active_streams[websocket] = False

        try:
            # Send welcome message
            await websocket.send_json({
                "type": "connected",
                "message": "Poker HUD ready. Send frames or use manual input.",
                "ocr_available": poker_ocr.available
            })

            # Handle messages
            while True:
                try:
                    data = await websocket.receive_json()
                    await self._handle_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON"
                    })

        except WebSocketDisconnect:
            logger.info("Poker stream client disconnected")
        except Exception as e:
            logger.error(f"Poker stream error: {e}")
        finally:
            self.connections.discard(websocket)
            self.active_streams.pop(websocket, None)

    async def _handle_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        msg_type = data.get("type", "")

        if msg_type == "start_stream":
            self.active_streams[websocket] = True
            await websocket.send_json({
                "type": "stream_started",
                "message": "Send webcam frames as base64",
                "fps": 5
            })

        elif msg_type == "stop_stream":
            self.active_streams[websocket] = False
            await websocket.send_json({
                "type": "stream_stopped",
                "message": "Webcam OCR stopped"
            })

        elif msg_type == "frame":
            # Process webcam frame with OCR
            image_b64 = data.get("image", "")
            if image_b64:
                await self._process_frame(websocket, image_b64)
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "No image data in frame"
                })

        elif msg_type == "manual_input":
            # Manual poker input (for testing or when OCR fails)
            await self._process_manual_input(websocket, data)

        elif msg_type == "vpip_action":
            # Track VPIP
            action = data.get("action", "fold")
            hole_cards = data.get("hole_cards", [])
            is_raise = data.get("is_raise", action == "raise")
            poker_ocr.update_vpip(hole_cards, action, is_raise)

            # Send updated stats
            stats = poker_ocr.get_vpip_stats()
            await websocket.send_json({
                "type": "vpip_updated",
                "action": action,
                "is_raise": is_raise
            })
            await websocket.send_json({
                "type": "hero_stats",
                "data": stats
            })

        elif msg_type == "get_vpip_stats":
            # Return VPIP statistics
            stats = poker_ocr.get_vpip_stats()
            await websocket.send_json({
                "type": "vpip_stats",
                "data": stats
            })
            await websocket.send_json({
                "type": "hero_stats",
                "data": stats
            })

        elif msg_type == "reset_session":
            # Reset session stats
            final_stats = poker_ocr.reset_session()
            await websocket.send_json({
                "type": "session_reset",
                "data": final_stats,
                "message": "Session reset. Stats cleared."
            })
            # Send fresh hero stats
            await websocket.send_json({
                "type": "hero_stats",
                "data": poker_ocr.get_vpip_stats()
            })

        elif msg_type == "get_session_summary":
            # Return full session summary
            summary = poker_ocr.get_session_summary()
            await websocket.send_json({
                "type": "session_summary",
                "data": summary
            })

        elif msg_type == "ping":
            await websocket.send_json({"type": "pong"})

        elif msg_type == "villain_action":
            # Track villain action
            await self._handle_villain_action(websocket, data)

        elif msg_type == "get_villain_stats":
            # Return all villain profiles
            profiles = villain_tracker.get_hud_data()
            dynamics = villain_tracker.analyze_table_dynamics()
            await websocket.send_json({
                "type": "villain_bulk",
                "profiles": {p["seat"]: p for p in profiles}
            })
            await websocket.send_json({
                "type": "table_dynamics",
                "data": dynamics
            })

        elif msg_type == "get_news":
            # Fetch news headlines for ticker
            count = data.get("count", 5)
            await self._handle_news_request(websocket, count)

        elif msg_type == "set_poker_site":
            # Switch poker site OCR mode
            site = data.get("site", "generic")
            result = poker_router.set_site(site)
            await websocket.send_json({
                "type": "site_changed",
                "site": poker_router.current_site.value,
                "message": result
            })

        elif msg_type == "refresh_meta":
            # Refresh poker meta trends
            force = data.get("force", False)
            await self._handle_meta_refresh(websocket, force)

        elif msg_type == "get_meta":
            # Get current meta snapshot
            await self._handle_get_meta(websocket)

        # Multi-table management
        elif msg_type == "register_table":
            table_id = data.get("table_id")
            site = data.get("site", "generic")
            bounds = data.get("window_bounds")
            new_id = table_manager.register_table(table_id, site, bounds)
            await websocket.send_json({
                "type": "table_registered",
                "table_id": new_id,
                "total_tables": table_manager.get_table_count()
            })

        elif msg_type == "remove_table":
            table_id = data.get("table_id")
            if table_id:
                table_manager.remove_table(table_id)
            await websocket.send_json({
                "type": "table_removed",
                "table_id": table_id,
                "total_tables": table_manager.get_table_count()
            })

        elif msg_type == "switch_table":
            table_id = data.get("table_id")
            if table_id and table_manager.set_active_table(table_id):
                table = table_manager.get_active_table()
                await websocket.send_json({
                    "type": "active_table_changed",
                    "table": table.to_dict() if table else None
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Table not found: {table_id}"
                })

        elif msg_type == "next_table":
            new_active = table_manager.next_table()
            table = table_manager.get_active_table()
            await websocket.send_json({
                "type": "active_table_changed",
                "table": table.to_dict() if table else None
            })

        elif msg_type == "prev_table":
            new_active = table_manager.prev_table()
            table = table_manager.get_active_table()
            await websocket.send_json({
                "type": "active_table_changed",
                "table": table.to_dict() if table else None
            })

        elif msg_type == "get_tables":
            await websocket.send_json({
                "type": "tables_list",
                "tables": table_manager.get_all_tables(),
                "active_table": table_manager.active_table,
                "total": table_manager.get_table_count()
            })

        # Live game mode (casino/home game support)
        elif msg_type == "live_mode_start":
            stakes = data.get("stakes", "1/2")
            table_size = data.get("table_size", 9)
            result = live_mode.activate(stakes, table_size)
            await websocket.send_json({
                "type": "live_mode_active",
                "data": result
            })

        elif msg_type == "live_mode_stop":
            result = live_mode.deactivate()
            await websocket.send_json({
                "type": "live_mode_ended",
                "data": result
            })

        elif msg_type == "live_new_hand":
            result = live_mode.new_hand()
            await websocket.send_json({
                "type": "live_hand_started",
                "data": result
            })

        elif msg_type == "live_set_cards":
            cards = data.get("cards", "")
            result = live_mode.set_hole_cards(cards)
            await websocket.send_json({
                "type": "live_cards_set",
                "data": result
            })
            # Send EV update if we have enough state
            if live_mode.state.hole_cards and live_mode.state.pot > 0:
                await self._send_live_ev(websocket)

        elif msg_type == "live_set_board":
            cards = data.get("cards", "")
            result = live_mode.set_board(cards)
            await websocket.send_json({
                "type": "live_board_set",
                "data": result
            })
            if live_mode.state.hole_cards and live_mode.state.pot > 0:
                await self._send_live_ev(websocket)

        elif msg_type == "live_add_card":
            card = data.get("card", "")
            result = live_mode.add_card(card)
            await websocket.send_json({
                "type": "live_card_added",
                "data": result
            })
            if live_mode.state.hole_cards and live_mode.state.pot > 0:
                await self._send_live_ev(websocket)

        elif msg_type == "live_set_pot":
            amount = float(data.get("amount", 0))
            result = live_mode.set_pot(amount)
            await websocket.send_json({
                "type": "live_pot_set",
                "data": result
            })

        elif msg_type == "live_set_stack":
            amount = float(data.get("amount", 0))
            result = live_mode.set_stack(amount)
            await websocket.send_json({
                "type": "live_stack_set",
                "data": result
            })

        elif msg_type == "live_set_position":
            position = data.get("position", "")
            result = live_mode.set_position(position)
            await websocket.send_json({
                "type": "live_position_set",
                "data": result
            })

        elif msg_type == "live_facing":
            action_type = data.get("action", "bet")
            amount = float(data.get("amount", 0))
            result = live_mode.facing_action(action_type, amount)
            await websocket.send_json({
                "type": "live_facing_action",
                "data": result
            })

        elif msg_type == "live_result":
            won = data.get("won", False)
            amount = float(data.get("amount", 0))
            result = live_mode.record_result(won, amount)
            await websocket.send_json({
                "type": "live_result_recorded",
                "data": result
            })

        elif msg_type == "get_live_state":
            await websocket.send_json({
                "type": "live_state",
                "data": live_mode.get_state()
            })

        elif msg_type == "live_command":
            # Process natural language command
            command = data.get("command", "")
            result = live_mode.process_voice_command(command)
            await websocket.send_json({
                "type": "live_command_result",
                "data": result
            })
            # Send EV update if state changed
            if live_mode.state.hole_cards and live_mode.state.pot > 0:
                await self._send_live_ev(websocket)

        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            })

    async def _process_frame(self, websocket: WebSocket, image_b64: str):
        """Process webcam frame with OCR and calculate EV."""
        try:
            # Run OCR using site router (supports Ignition mode)
            table_state = await poker_router.analyze_frame(image_b64)

            # Send table state
            await websocket.send_json({
                "type": "table_state",
                "data": table_state.to_dict()
            })

            # Check for new hand detection (for session tracking)
            is_new_hand = poker_ocr.detect_new_hand(table_state)
            if is_new_hand:
                # Send hero stats update on new hand
                stats = poker_ocr.get_vpip_stats()
                await websocket.send_json({
                    "type": "hero_stats",
                    "data": stats,
                    "new_hand": True,
                    "hole_cards": table_state.hole_cards
                })

            # Process showdown data if present (Ignition mode)
            showdown_data = getattr(table_state, 'showdown', None)
            if showdown_data:
                await self._process_showdown(websocket, showdown_data)

            # Calculate and send SPR if we have stack and pot
            if table_state.player_stack and table_state.pot_size > 0:
                spr_data = spr_calculator.calculate_spr(
                    table_state.player_stack,
                    table_state.pot_size
                )
                await websocket.send_json({
                    "type": "spr_update",
                    "data": spr_data
                })

            # If we have cards and pot, calculate EV
            if table_state.hole_cards and table_state.pot_size > 0:
                await self._calculate_and_send_ev(
                    websocket,
                    table_state.hole_cards,
                    table_state.community_cards,
                    table_state.pot_size,
                    table_state.current_bet,
                    stack=table_state.player_stack
                )

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"OCR error: {str(e)}"
            })

    async def _process_manual_input(self, websocket: WebSocket, data: Dict[str, Any]):
        """Process manual poker input."""
        hole_cards_str = data.get("hole_cards", "")
        board_str = data.get("board", "")
        pot = float(data.get("pot", 0))
        bet = float(data.get("bet", 0))
        opponents = int(data.get("opponents", 1))

        # Parse cards
        hole_cards = poker_ocr._parse_cards(hole_cards_str)
        community_cards = poker_ocr._parse_cards(board_str) if board_str else []

        if len(hole_cards) != 2:
            await websocket.send_json({
                "type": "error",
                "message": "Need exactly 2 hole cards (e.g., 'AhKd')"
            })
            return

        # Create table state
        table_state = PokerTableState(
            hole_cards=hole_cards,
            community_cards=community_cards,
            pot_size=pot,
            current_bet=bet,
            player_stack=None,
            street=self._get_street(community_cards),
            num_players=opponents + 1,
            position=None,
            raw_text="manual input",
            confidence=1.0,
            timestamp=datetime.utcnow()
        )

        await websocket.send_json({
            "type": "table_state",
            "data": table_state.to_dict()
        })

        # Calculate EV
        await self._calculate_and_send_ev(
            websocket,
            hole_cards,
            community_cards,
            pot,
            bet,
            opponents
        )

    async def _calculate_and_send_ev(
        self,
        websocket: WebSocket,
        hole_cards: list,
        community_cards: list,
        pot: float,
        bet: float,
        opponents: int = 1,
        stack: float = 0
    ):
        """Calculate and send EV table."""
        # Format cards for poker tool
        hand_str = "".join(hole_cards)
        board_str = "".join(community_cards) if community_cards else ""

        # Calculate equity using Monte Carlo
        equity_result = await self.poker_tool.execute(
            action="equity",
            hand=hand_str,
            board=board_str,
            opponents=opponents,
            simulations=5000  # Faster for real-time
        )

        equity = 50.0  # Default
        if equity_result.success:
            equity = equity_result.data.get("equity", 50.0)

        # Calculate EV table for different bet sizes
        ev_table = ev_calculator.calculate_ev_table(
            pot=pot,
            equity=equity,
            villain_fold_pct=30  # Assume 30% fold to bets
        )

        # Calculate pot odds if there's a bet
        pot_odds_required = 0
        if bet > 0:
            pot_odds_required = ev_calculator.pot_odds_required(bet, pot)

        # Calculate SPR if we have stack info
        spr_data = None
        if stack and stack > 0 and pot > 0:
            spr_data = spr_calculator.calculate_spr(stack, pot)

        # Build response
        response = {
            "type": "ev_table",
            "data": {
                "hand": hand_str,
                "board": board_str,
                "equity": round(equity, 1),
                "pot": pot,
                "bet_to_call": bet,
                "pot_odds_required": round(pot_odds_required, 1),
                "should_call": equity > pot_odds_required if bet > 0 else None,
                "ev_by_size": ev_table,
                "recommendation": self._get_recommendation(equity, pot_odds_required, bet),
                "spr": spr_data
            }
        }

        await websocket.send_json(response)

    def _get_street(self, community_cards: list) -> str:
        """Determine street from community cards."""
        if len(community_cards) == 0:
            return "preflop"
        elif len(community_cards) == 3:
            return "flop"
        elif len(community_cards) == 4:
            return "turn"
        else:
            return "river"

    def _get_recommendation(self, equity: float, pot_odds_required: float, bet: float) -> str:
        """Generate action recommendation."""
        if bet == 0:
            if equity > 65:
                return "Strong hand - consider betting for value"
            elif equity > 45:
                return "Marginal hand - check or small bet"
            else:
                return "Weak hand - check and evaluate"

        # There's a bet to call
        edge = equity - pot_odds_required

        if edge > 15:
            return f"Clear CALL or RAISE. {edge:.0f}% edge over pot odds."
        elif edge > 5:
            return f"Profitable CALL. {edge:.0f}% edge."
        elif edge > -5:
            return f"Borderline. Consider implied odds and reads."
        else:
            return f"FOLD. {-edge:.0f}% below required equity."

    async def _handle_villain_action(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle villain action tracking."""
        from backend.poker.villain_tracker import Street, PokerAction

        seat = data.get("seat")
        if not seat:
            await websocket.send_json({
                "type": "error",
                "message": "Villain action requires 'seat' field"
            })
            return

        # Map string action to PokerAction enum
        action_map = {
            "fold": PokerAction.FOLD,
            "check": PokerAction.CHECK,
            "call": PokerAction.CALL,
            "bet": PokerAction.BET,
            "raise": PokerAction.RAISE,
            "all_in": PokerAction.ALL_IN,
            "allin": PokerAction.ALL_IN
        }
        action_str = data.get("action", "fold").lower()
        action = action_map.get(action_str, PokerAction.FOLD)

        # Map string street to Street enum
        street_map = {
            "preflop": Street.PREFLOP,
            "flop": Street.FLOP,
            "turn": Street.TURN,
            "river": Street.RIVER
        }
        street_str = data.get("street", "preflop").lower()
        street = street_map.get(street_str, Street.PREFLOP)

        # Record the action
        villain_tracker.record_action(
            seat=seat,
            street=street,
            action=action,
            amount=data.get("amount", 0.0),
            pot_size=data.get("pot_size", 0.0),
            is_preflop_voluntary=data.get("is_voluntary", False),
            is_preflop_raise=data.get("is_raise", False),
            facing_cbet=data.get("facing_cbet", False),
            facing_3bet=data.get("facing_3bet", False),
            is_3bet=data.get("is_3bet", False)
        )

        # Send updated profile
        profile = villain_tracker.get_profile(seat)
        if profile:
            await websocket.send_json({
                "type": "villain_stats",
                "seat": seat,
                "data": profile.to_hud_format()
            })

    async def _process_showdown(self, websocket: WebSocket, showdown_str: str):
        """Process showdown data from OCR.

        Args:
            websocket: WebSocket connection
            showdown_str: Showdown string (e.g., "Seat 3|Two Pair|Ac Kh")
        """
        try:
            # Parse showdown using hand history parser
            showdown = hand_history_parser.parse_ignition_format(showdown_str)

            if showdown:
                # Record in hand history
                hand_history_parser.record_showdown(showdown)

                # Update villain tracker
                villain_tracker.record_showdown(
                    seat=showdown.seat,
                    cards=showdown.cards,
                    hand_description=showdown.hand_description
                )

                # Send showdown notification
                await websocket.send_json({
                    "type": "showdown_detected",
                    "data": showdown.to_dict()
                })

                # Send updated villain stats
                profile = villain_tracker.get_profile(showdown.seat)
                if profile:
                    await websocket.send_json({
                        "type": "villain_stats",
                        "seat": showdown.seat,
                        "data": profile.to_hud_format()
                    })

                logger.info(f"Processed showdown: {showdown.seat} showed {showdown.cards}")

        except Exception as e:
            logger.error(f"Showdown processing error: {e}")

    async def _handle_news_request(self, websocket: WebSocket, count: int = 5):
        """Handle news ticker request."""
        try:
            news_tool = get_news_tool()
            headlines = await news_tool.get_ticker_headlines(count)
            await websocket.send_json({
                "type": "news_headlines",
                "headlines": headlines
            })
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            await websocket.send_json({
                "type": "news_headlines",
                "headlines": []
            })

    async def _handle_meta_refresh(self, websocket: WebSocket, force: bool = False):
        """Handle meta trends refresh request."""
        try:
            meta_data = await ev_calculator.refresh_meta(force)
            await websocket.send_json({
                "type": "meta_updated",
                "data": meta_data,
                "message": "Meta trends refreshed" if not meta_data.get("error") else meta_data["error"]
            })
        except Exception as e:
            logger.error(f"Meta refresh error: {e}")
            await websocket.send_json({
                "type": "meta_updated",
                "data": {},
                "error": str(e)
            })

    async def _handle_get_meta(self, websocket: WebSocket):
        """Handle get current meta request."""
        try:
            from backend.poker.meta_advisor import meta_advisor
            meta_snapshot = meta_advisor.get_current_meta()
            await websocket.send_json({
                "type": "meta_snapshot",
                "data": meta_snapshot.to_dict()
            })
        except ImportError:
            await websocket.send_json({
                "type": "meta_snapshot",
                "data": {},
                "error": "Meta advisor not available"
            })
        except Exception as e:
            logger.error(f"Get meta error: {e}")
            await websocket.send_json({
                "type": "meta_snapshot",
                "data": {},
                "error": str(e)
            })

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        for ws in list(self.connections):
            try:
                await ws.send_json(message)
            except Exception:
                self.connections.discard(ws)

    async def broadcast_villain_update(self, seat: str):
        """Broadcast villain profile update to all clients."""
        profile = villain_tracker.get_profile(seat)
        if profile:
            await self.broadcast({
                "type": "villain_stats",
                "seat": seat,
                "data": profile.to_hud_format()
            })

    async def broadcast_news(self, count: int = 5):
        """Broadcast news headlines to all clients."""
        try:
            news_tool = get_news_tool()
            headlines = await news_tool.get_ticker_headlines(count)
            await self.broadcast({
                "type": "news_headlines",
                "headlines": headlines
            })
        except Exception as e:
            logger.error(f"News broadcast error: {e}")

    async def broadcast_hero_stats(self):
        """Broadcast hero session stats to all clients."""
        stats = poker_ocr.get_vpip_stats()
        await self.broadcast({
            "type": "hero_stats",
            "data": stats
        })

    async def _send_live_ev(self, websocket: WebSocket):
        """Send EV calculation for live game state."""
        try:
            engine_state = live_mode.get_engine_state()
            if not engine_state:
                return

            hole_cards = engine_state.get("hole_cards", [])
            community = engine_state.get("community_cards", [])
            pot = engine_state.get("pot_size", 0)
            stack = engine_state.get("player_stack", 0)
            bet_to_call = engine_state.get("bet_to_call", 0)
            players = engine_state.get("players_in_hand", 2)

            if not hole_cards or len(hole_cards) < 2:
                return

            # Calculate equity
            hand_str = "".join(hole_cards)
            board_str = "".join(community) if community else ""

            equity_result = await self.poker_tool.execute(
                action="equity",
                hand=hand_str,
                board=board_str,
                opponents=max(1, players - 1),
                simulations=3000
            )

            equity = 50.0
            if equity_result.success:
                equity = equity_result.data.get("equity", 50.0)

            # Calculate EV table
            ev_table = ev_calculator.calculate_ev_table(
                pot=pot,
                equity=equity,
                villain_fold_pct=30
            )

            # Calculate pot odds if facing a bet
            pot_odds_required = 0
            if bet_to_call > 0:
                pot_odds_required = ev_calculator.pot_odds_required(bet_to_call, pot)

            # Calculate SPR
            spr_data = None
            if stack > 0 and pot > 0:
                spr_data = spr_calculator.calculate_spr(stack, pot)

            # Build and send response
            await websocket.send_json({
                "type": "live_ev_update",
                "data": {
                    "hand": hand_str,
                    "board": board_str,
                    "equity": round(equity, 1),
                    "pot": pot,
                    "stack": stack,
                    "bet_to_call": bet_to_call,
                    "pot_odds_required": round(pot_odds_required, 1),
                    "should_call": equity > pot_odds_required if bet_to_call > 0 else None,
                    "ev_by_size": ev_table,
                    "spr": spr_data,
                    "position": live_mode.state.get_position_name(),
                    "street": live_mode.state.get_street_name(),
                    "recommendation": self._get_recommendation(equity, pot_odds_required, bet_to_call)
                }
            })

        except Exception as e:
            logger.error(f"Live EV calculation error: {e}")

    async def broadcast_live_state(self):
        """Broadcast live mode state to all clients."""
        if live_mode.is_active:
            await self.broadcast({
                "type": "live_state",
                "data": live_mode.get_state()
            })


# Global handler instance
poker_stream_handler = PokerStreamHandler()
