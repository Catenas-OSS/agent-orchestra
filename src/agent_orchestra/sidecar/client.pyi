from typing import  Dict

# This is a stub file to define the MCPClient interface
class MCPSession:
    """Stub for MCPSession class"""
    pass

class MCPClient:
    """Stub for MCPClient class"""
    
    sessions: Dict[str, MCPSession]
    
    async def create_session(self, server_name: str, auto_initialize: bool = True) -> MCPSession:
        """Create a session for a server"""
        ...
        
    async def create_all_sessions(self, auto_initialize: bool = True) -> Dict[str, MCPSession]:
        """Create sessions for all configured servers"""
        ...