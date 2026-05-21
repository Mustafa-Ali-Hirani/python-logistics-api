# This simulates an external MCP Server providing a "Resource"
class MockMCPServer:
    def __init__(self):
        # This is our 'External Database'
        self.container_registry = {
            "ABC-123": {"location": "Dock A", "status": "Cleared"},
            "XYZ-999": {"location": "Warehouse B", "status": "In Inspection"},
            "LOG-777": {"location": "On Ship", "status": "Departed"}
        }

    def get_container_info(self, container_id: str):
        """Standardized tool provided by the MCP Server"""
        return self.container_registry.get(container_id, "Container ID not found in registry.")

    def list_all_resources(self):
        """The 'Protocol' way to show what data we have available"""
        return list(self.container_registry.keys())