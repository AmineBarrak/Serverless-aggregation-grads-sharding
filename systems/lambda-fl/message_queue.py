"""Simulated message queue (Kafka-like) for Lambda-FL.

Implements thread-safe publish/subscribe communication between parties and aggregators.
Uses Python Queue and dict-based storage to simulate Kafka topics.
"""

from typing import Any, Callable, Dict, List, Optional
from queue import Queue
import threading
import logging


logger = logging.getLogger(__name__)


class SimulatedKafka:
    """Simulates Kafka message broker for federated learning communication.

    Topics are dynamically created on first publish/subscribe.
    Thread-safe with locks for all operations.
    """

    def __init__(self):
        """Initialize the message broker."""
        self._topics: Dict[str, Queue] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()

    def publish(self, topic: str, message: Any) -> None:
        """Publish a message to a topic.

        Args:
            topic: Topic name to publish to
            message: Message payload (any Python object)
        """
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = Queue()
                self._subscribers[topic] = []

            self._topics[topic].put(message)
            logger.debug(f"Published message to {topic}")

            # Call all subscribers for this topic
            for callback in self._subscribers[topic]:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in subscriber callback for {topic}: {e}")

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """Subscribe a callback to a topic.

        Args:
            topic: Topic name to subscribe to
            callback: Function to call when messages arrive
        """
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = Queue()
                self._subscribers[topic] = []

            self._subscribers[topic].append(callback)
            logger.debug(f"Subscribed callback to {topic}")

    def consume(self, topic: str, count: int, timeout: Optional[float] = None) -> List[Any]:
        """Consume messages from a topic.

        Blocks until count messages are available or timeout expires.

        Args:
            topic: Topic name to consume from
            count: Number of messages to consume
            timeout: Max time to wait for all messages (seconds). None means wait indefinitely.

        Returns:
            List of messages (may be shorter than count if timeout occurs)
        """
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = Queue()
                self._subscribers[topic] = []

        messages = []
        try:
            for _ in range(count):
                msg = self._topics[topic].get(timeout=timeout)
                messages.append(msg)
        except Exception as e:
            logger.debug(f"Timeout or error consuming from {topic}: {e}")

        return messages

    def get_topic_size(self, topic: str) -> int:
        """Get current size of a topic's message queue.

        Args:
            topic: Topic name

        Returns:
            Number of messages in queue
        """
        with self._lock:
            if topic not in self._topics:
                return 0
            return self._topics[topic].qsize()

    def clear_topic(self, topic: str) -> None:
        """Clear all messages from a topic.

        Args:
            topic: Topic name to clear
        """
        with self._lock:
            if topic in self._topics:
                while not self._topics[topic].empty():
                    try:
                        self._topics[topic].get_nowait()
                    except Exception:
                        break
            logger.debug(f"Cleared topic {topic}")
