"""
Tests for playing a podcast as a whole, without a start episode.

The queue starts at the newest episode and works back through the older ones,
whatever order the provider lists them in and whatever was already played.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import Podcast, PodcastEpisode, ProviderMapping

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant

PROVIDER = "test_podcast_prov"


def _provider_mapping() -> set[ProviderMapping]:
    """Create a single provider mapping with a unique item id."""
    return {
        ProviderMapping(
            item_id=uuid4().hex,
            provider_domain=PROVIDER,
            provider_instance=PROVIDER,
        )
    }


def _podcast() -> Podcast:
    """Create a podcast to resolve episodes for."""
    return Podcast(
        item_id="show-order-001",
        provider=PROVIDER,
        name="Default Order Show",
        provider_mappings=_provider_mapping(),
    )


def _episode(podcast: Podcast, number: int, position: int) -> PodcastEpisode:
    """Create an episode carrying the given position."""
    return PodcastEpisode(
        item_id=f"ep-order-{number:03d}",
        provider=PROVIDER,
        name=f"Episode {number}",
        provider_mappings=_provider_mapping(),
        position=position,
        podcast=podcast,
    )


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    mass: MusicAssistant,
    episodes: list[PodcastEpisode],
    resume: tuple[bool, int] = (False, 0),
) -> None:
    """
    Let the podcasts controller list the given episodes, in the given order.

    :param resume: The ``(fully_played, resume_position_ms)`` stored for every episode.
    """

    async def _episodes(_item_id: str, _provider: str) -> AsyncGenerator[PodcastEpisode]:
        for episode in episodes:
            yield episode

    async def _get_resume_position(*_args: object, **_kwargs: object) -> tuple[bool, int]:
        return resume

    monkeypatch.setattr(mass.music.podcasts, "episodes", _episodes)
    monkeypatch.setattr(mass.music, "get_resume_position", _get_resume_position)


@pytest.mark.parametrize("listing", [(1, 2, 3), (3, 2, 1), (2, 3, 1)])
async def test_podcast_queues_newest_episode_first(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch, listing: tuple[int, ...]
) -> None:
    """The whole podcast is queued newest to oldest, whatever order the provider lists."""
    podcast = _podcast()
    _serve(monkeypatch, mass, [_episode(podcast, number, position=number) for number in listing])

    resolved = await mass.player_queues._media_resolver.get_next_podcast_episodes(podcast, None)

    assert [x.item_id for x in resolved] == ["ep-order-003", "ep-order-002", "ep-order-001"]


async def test_podcast_starts_at_newest_episode_even_when_fully_played(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished newest episode is played again from its start, not skipped."""
    podcast = _podcast()
    episodes = [_episode(podcast, number, position=number) for number in (1, 2, 3)]
    _serve(monkeypatch, mass, episodes, resume=(True, 90_000))

    resolved = await mass.player_queues._media_resolver.get_next_podcast_episodes(podcast, None)

    assert resolved[0].item_id == "ep-order-003"
    assert resolved[0].resume_position_ms == 0


async def test_podcast_resumes_a_partly_played_newest_episode(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The newest episode continues where it was left, unless asked to start over."""
    podcast = _podcast()
    episodes = [_episode(podcast, number, position=number) for number in (1, 2, 3)]
    _serve(monkeypatch, mass, episodes, resume=(False, 90_000))
    resolver = mass.player_queues._media_resolver

    resumed = await resolver.get_next_podcast_episodes(podcast, None)
    assert resumed[0].item_id == "ep-order-003"
    assert resumed[0].resume_position_ms == 90_000

    from_start = await resolver.get_next_podcast_episodes(podcast, None, start_from_beginning=True)
    assert from_start[0].item_id == "ep-order-003"
    assert from_start[0].resume_position_ms == 0


async def test_podcast_without_episodes_cannot_be_played(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A podcast with an empty feed raises instead of returning an empty queue."""
    _serve(monkeypatch, mass, [])

    with pytest.raises(InvalidDataError):
        await mass.player_queues._media_resolver.get_next_podcast_episodes(_podcast(), None)
