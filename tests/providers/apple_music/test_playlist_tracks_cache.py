"""Unit tests for keeping Apple Music playlist tracks cached ahead of playback."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from music_assistant_models.background_task import TaskScheduleType
from music_assistant_models.errors import ResourceTemporarilyUnavailable

from music_assistant.controllers.cache.controller import CacheController
from music_assistant.controllers.cache.helpers import BYPASS_CACHE
from music_assistant.providers.apple_music.constants import (
    PLAYLIST_TRACKS_REFRESH_INTERVAL_HOURS,
    PLAYLIST_TRACKS_REFRESH_TASK_ID,
)
from music_assistant.providers.apple_music.media import AppleMusicMediaManager
from music_assistant.providers.apple_music.provider import AppleMusicProvider
from tests.common import use_real_create_task


def _make_manager(
    cache_lookup: Any, get_data: Any
) -> tuple[AppleMusicMediaManager, MagicMock, MagicMock]:
    """
    Build a media manager on a mocked provider.

    :param cache_lookup: Callable returning the ``(data, is_fresh, found)`` cache lookup result.
    :param get_data: Stand-in for the API client's ``get_data``.
    """
    provider = MagicMock()
    provider.domain = "apple_music"
    provider.instance_id = "apple_music--test"
    provider._storefront = "us"
    cache = provider.mass.cache

    async def _get_with_freshness(_key: str, **kwargs: Any) -> tuple[Any, bool, bool]:
        return cache_lookup(**kwargs)  # type: ignore[no-any-return]

    cache.get_with_freshness = _get_with_freshness
    cache.set = AsyncMock()
    cache.handle_refresh = lambda bypass: CacheController.handle_refresh(cache, bypass)
    use_real_create_task(provider.mass)
    api = provider.api_client
    api.get_data = AsyncMock(side_effect=get_data)
    api.get_ratings = AsyncMock(return_value={})
    return AppleMusicMediaManager(provider), api, cache


async def _drain(mass: MagicMock) -> None:
    """Wait for the background tasks (cache refreshes and writes) to finish."""
    while pending := [task for task in mass._tracked_tasks.values() if not task.done()]:
        await asyncio.gather(*pending)


async def test_expired_playlist_tracks_do_not_block_playback() -> None:
    """An expired listing is returned as-is while the re-fetch happens in the background."""
    release_api = asyncio.Event()

    async def _get_data(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        # Apple throttling the request for as long as it likes
        await release_api.wait()
        return {}

    def _expired_entry(include_expired: bool = False, **_kwargs: Any) -> tuple[Any, bool, bool]:
        return ([], False, True) if include_expired else (None, False, False)

    manager, api, cache = _make_manager(_expired_entry, _get_data)

    tracks = await asyncio.wait_for(manager.get_playlist_tracks("p.test", page=0), timeout=1)

    assert tracks == []
    cache.set.assert_not_awaited()
    release_api.set()
    await _drain(manager.mass)
    api.get_data.assert_awaited_once()
    stored = cache.set.await_args.kwargs
    assert stored["allow_expired_cache"] is True
    # the scheduled refresh has to come round before an entry expires
    assert stored["expiration"] > PLAYLIST_TRACKS_REFRESH_INTERVAL_HOURS * 3600


async def test_refresh_task_refetches_every_page_of_every_library_playlist() -> None:
    """The refresh bypasses a still fresh cache and survives a playlist that fails."""
    pages = {"p.big": 2, "p.broken": None, "p.small": 1}  # pages holding tracks
    fetched: list[tuple[str, int]] = []

    async def _get_data(endpoint: str, **kwargs: Any) -> dict[str, Any]:
        playlist_id = endpoint.split("/")[-2]
        page = kwargs["offset"] // 100
        fetched.append((playlist_id, page))
        page_count = pages[playlist_id]
        if page_count is None:
            raise ResourceTemporarilyUnavailable("Apple Music Rate Limiter")
        return {"data": [None]} if page < page_count else {}

    def _fresh_entry(**_kwargs: Any) -> tuple[Any, bool, bool]:
        return (None, False, False) if BYPASS_CACHE.get() else ([], True, True)

    manager, _, cache = _make_manager(_fresh_entry, _get_data)
    fetch_page = manager._get_playlist_tracks_cached

    async def _get_playlist_tracks(playlist_id: str, page: int = 0) -> bool:
        # the listing rows are not parsable tracks, so report "more pages" from the fixture
        await fetch_page(playlist_id, page)
        return page < (pages[playlist_id] or 0)

    async def _get_library_playlists() -> Any:
        for playlist_id in pages:
            yield SimpleNamespace(item_id=playlist_id, name=playlist_id)

    provider = SimpleNamespace(
        mass=manager.mass,
        logger=MagicMock(),
        get_library_playlists=_get_library_playlists,
        get_playlist_tracks=_get_playlist_tracks,
    )

    await AppleMusicProvider._refresh_playlist_tracks(provider)  # type: ignore[arg-type]
    await _drain(manager.mass)

    assert fetched == [
        ("p.big", 0),
        ("p.big", 1),
        ("p.big", 2),
        ("p.broken", 0),
        ("p.small", 0),
        ("p.small", 1),
    ]
    # every page is written back, including the empty one that ends each listing
    stored_keys = sorted(call.kwargs["key"] for call in cache.set.await_args_list)
    assert stored_keys == [
        f"_get_playlist_tracks_cached.{playlist_id}.{page}"
        for playlist_id, page in fetched
        if playlist_id != "p.broken"
    ]
    provider.logger.warning.assert_called_once()
    assert not BYPASS_CACHE.get()


async def test_refresh_task_is_scheduled_on_init_and_removed_on_unload() -> None:
    """The provider registers the recurring refresh and unregisters it again."""
    provider = MagicMock(spec=AppleMusicProvider)
    provider.mass = MagicMock()
    provider.instance_id = "apple_music--test"
    provider.api_client = MagicMock(get_user_storefront=AsyncMock(return_value="us"))
    provider.streaming_manager = MagicMock(initialize=AsyncMock())
    provider._recommendation_payload_task = None
    provider._recommendation_refresh_task = None
    task_id = f"{PLAYLIST_TRACKS_REFRESH_TASK_ID}_apple_music--test"

    await AppleMusicProvider.handle_async_init(provider)

    registered = provider.mass.tasks.register_scheduled_task.call_args.kwargs
    assert registered["task_id"] == task_id
    assert registered["handler"] == provider._refresh_playlist_tracks
    assert registered["schedule"].type == TaskScheduleType.HOURLY
    assert registered["schedule"].every == PLAYLIST_TRACKS_REFRESH_INTERVAL_HOURS

    await AppleMusicProvider.unload(provider)

    provider.mass.tasks.unregister_scheduled_task.assert_called_once_with(
        task_id, clear_persisted_state=False
    )
