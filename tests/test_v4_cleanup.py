import asyncio
from types import SimpleNamespace

import pytest
from modal_proto import api_pb2

from decision_lab.studies.v4.cleanup import verify_owned_apps


class FakeStub:
    def __init__(self, state=api_pb2.APP_STATE_STOPPED, tasks=()):
        self.state = state
        self.tasks = tasks
        self.lifecycle_requests = []

    async def AppGetLifecycle(self, request):
        self.lifecycle_requests.append(request.app_id)
        return SimpleNamespace(lifecycle=SimpleNamespace(app_state=self.state, stopped_at=1.0))

    async def TaskList(self, request):
        return SimpleNamespace(tasks=self.tasks)


def test_verifier_queries_every_owned_id_without_recent_app_listing():
    stub = FakeStub()
    ids = ["ap-old-diagnostic", "ap-old-pilot", "ap-recent-test"]
    rows = asyncio.run(verify_owned_apps(SimpleNamespace(_stub=stub), ids))
    assert stub.lifecycle_requests == ids
    assert [row["app_id"] for row in rows] == ids
    assert all(row["state"] == "stopped" and row["tasks"] == 0 for row in rows)


@pytest.mark.parametrize(
    ("state", "tasks"), [(api_pb2.APP_STATE_DEPLOYED, ()), (api_pb2.APP_STATE_STOPPED, (object(),))]
)
def test_verifier_rejects_active_app_or_remaining_task(state, tasks):
    client = SimpleNamespace(_stub=FakeStub(state, tasks))
    with pytest.raises(ValueError, match="Owned app cleanup is not verified"):
        asyncio.run(verify_owned_apps(client, ["ap-owned"]))
