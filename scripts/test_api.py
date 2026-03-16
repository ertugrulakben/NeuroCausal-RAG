"""
NeuroCausal RAG - API Test Script
Tests FastAPI endpoints and validates Feedback Loop

Usage:
    # First start the API:
    uvicorn neurocausal_rag.api.app:app --reload

    # Then run the test:
    python scripts/test_api.py

    # Or with Docker:
    docker-compose up -d api
    python scripts/test_api.py --host localhost --port 8000

Author: Ertugrul Akben
"""

import requests
import json
import time
import argparse
from datetime import datetime


class APITester:
    """API Test Runner"""

    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key

        self.results = []

    def log(self, message: str, status: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {"INFO": "ℹ️", "OK": "✅", "FAIL": "❌", "WARN": "⚠️"}
        icon = icons.get(status, "•")
        print(f"[{timestamp}] {icon} {message}")

    def test_endpoint(self, method: str, path: str, data: dict = None, expected_status: int = 200):
        """Test a single endpoint"""
        url = f"{self.base_url}{path}"

        try:
            start = time.time()

            if method == "GET":
                response = requests.get(url, headers=self.headers)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=self.headers)
            else:
                raise ValueError(f"Unknown method: {method}")

            elapsed = (time.time() - start) * 1000

            success = response.status_code == expected_status

            result = {
                "endpoint": f"{method} {path}",
                "status_code": response.status_code,
                "expected": expected_status,
                "success": success,
                "time_ms": elapsed,
                "response": response.json() if response.text else None
            }

            self.results.append(result)

            if success:
                self.log(f"{method} {path} -> {response.status_code} ({elapsed:.1f}ms)", "OK")
            else:
                self.log(f"{method} {path} -> {response.status_code} (expected {expected_status})", "FAIL")

            return result

        except requests.exceptions.ConnectionError:
            self.log(f"{method} {path} -> Connection refused", "FAIL")
            return {"success": False, "error": "Connection refused"}
        except Exception as e:
            self.log(f"{method} {path} -> Error: {e}", "FAIL")
            return {"success": False, "error": str(e)}

    def run_all_tests(self):
        """Run all API tests"""
        print("\n" + "=" * 60)
        print("NEUROCAUSAL RAG - API TEST")
        print("=" * 60)
        print(f"Base URL: {self.base_url}")
        print(f"API Key: {'***' + self.api_key[-4:] if self.api_key else 'None (auth disabled)'}")
        print("=" * 60 + "\n")

        # =================================================================
        # TEST 1: Health Check
        # =================================================================
        self.log("Testing Health Check...", "INFO")
        health = self.test_endpoint("GET", "/api/v1/health")

        if not health.get("success"):
            self.log("API is not running! Start it with: uvicorn neurocausal_rag.api.app:app --reload", "FAIL")
            return

        # =================================================================
        # TEST 2: Root Endpoint
        # =================================================================
        self.log("Testing Root Endpoint...", "INFO")
        self.test_endpoint("GET", "/")

        # =================================================================
        # TEST 3: Add Documents
        # =================================================================
        self.log("Testing Document Creation...", "INFO")

        docs = [
            {"id": "test_stress", "content": "Stres, kortizol hormonu salgilanmasina neden olur."},
            {"id": "test_kortizol", "content": "Yuksek kortizol, uyku duzenini bozar."},
            {"id": "test_uyku", "content": "Uykusuzluk, dikkat daginikligina yol acar."},
            {"id": "test_dikkat", "content": "Dikkat daginikligi is kazasi riskini artirir."},
        ]

        for doc in docs:
            self.test_endpoint("POST", "/api/v1/documents", doc, 201)

        # =================================================================
        # TEST 4: Search
        # =================================================================
        self.log("Testing Search Endpoint...", "INFO")

        search_result = self.test_endpoint("POST", "/api/v1/search", {
            "query": "Stres is kazalarini nasil etkiler?",
            "top_k": 5,
            "mode": "balanced"
        })

        if search_result.get("success"):
            response = search_result.get("response", {})
            results = response.get("results", [])
            self.log(f"Search returned {len(results)} results", "INFO")

            # Show results
            for i, r in enumerate(results[:3], 1):
                self.log(f"  Result {i}: {r.get('id')} (score: {r.get('score', 0):.3f})", "INFO")

        # =================================================================
        # TEST 5: Agent Query
        # =================================================================
        self.log("Testing Agent Endpoint...", "INFO")

        agent_result = self.test_endpoint("POST", "/api/v1/agent/query", {
            "query": "Stres nasil is kazalarina yol acar?",
            "max_iterations": 2,
            "min_confidence": 0.5
        })

        if agent_result.get("success"):
            response = agent_result.get("response", {})
            self.log(f"Agent confidence: {response.get('confidence', 0):.2f}", "INFO")
            self.log(f"Agent iterations: {response.get('iterations', 0)}", "INFO")

        # =================================================================
        # TEST 6: Feedback (RLHF)
        # =================================================================
        self.log("Testing Feedback Endpoint (RLHF)...", "INFO")

        # Positive feedback
        feedback_result = self.test_endpoint("POST", "/api/v1/feedback", {
            "query": "stres etkileri",
            "result_ids": ["test_stress", "test_kortizol"],
            "rating": 0.9,
            "comment": "Cok faydali sonuclar!"
        }, 201)

        if feedback_result.get("success"):
            self.log("Positive feedback recorded", "OK")

        # Negative feedback
        self.test_endpoint("POST", "/api/v1/feedback", {
            "query": "alakasiz sorgu",
            "result_ids": ["test_dikkat"],
            "rating": 0.2,
            "comment": "Ilgisiz sonuc"
        }, 201)

        # =================================================================
        # TEST 7: Graph Stats
        # =================================================================
        self.log("Testing Graph Stats...", "INFO")

        stats_result = self.test_endpoint("GET", "/api/v1/graph/stats")

        if stats_result.get("success"):
            response = stats_result.get("response", {})
            self.log(f"Nodes: {response.get('total_nodes', 0)}", "INFO")
            self.log(f"Edges: {response.get('total_edges', 0)}", "INFO")

        # =================================================================
        # TEST 8: Causal Chain
        # =================================================================
        self.log("Testing Causal Chain...", "INFO")

        self.test_endpoint("POST", "/api/v1/graph/chain", {
            "node_id": "test_stress",
            "max_depth": 3,
            "direction": "forward"
        })

        # =================================================================
        # TEST 9: Metrics
        # =================================================================
        self.log("Testing Metrics Endpoint...", "INFO")

        metrics_result = self.test_endpoint("GET", "/api/v1/metrics")

        if metrics_result.get("success"):
            response = metrics_result.get("response", {})
            self.log(f"Total requests: {response.get('requests_total', 0)}", "INFO")
            self.log(f"Avg response time: {response.get('avg_response_time_ms', 0):.1f}ms", "INFO")

        # =================================================================
        # TEST 10: Discovery
        # =================================================================
        self.log("Testing Discovery Endpoint...", "INFO")

        self.test_endpoint("POST", "/api/v1/discovery", {
            "mode": "fast",
            "min_confidence": 0.5,
            "max_relations": 10
        })

        # =================================================================
        # SUMMARY
        # =================================================================
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("success"))
        failed = total - passed

        print(f"\nTotal tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")

        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.get("success"):
                    print(f"  - {r.get('endpoint')}: {r.get('error', r.get('status_code'))}")

        avg_time = sum(r.get("time_ms", 0) for r in self.results) / max(1, total)
        print(f"\nAverage response time: {avg_time:.1f}ms")

        print("\n" + "=" * 60)

        if failed == 0:
            print("ALL TESTS PASSED! API is working correctly.")
        else:
            print(f"WARNING: {failed} test(s) failed. Check the logs above.")

        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NeuroCausal RAG API Tester")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", default="8000", help="API port")
    parser.add_argument("--api-key", default=None, help="API key (optional)")

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    tester = APITester(base_url, args.api_key)
    tester.run_all_tests()


if __name__ == "__main__":
    main()
