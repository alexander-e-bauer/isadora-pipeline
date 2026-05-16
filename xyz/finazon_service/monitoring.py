# Enhanced monitoring.py - Complete implementation
import time
import psutil
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import json
from collections import defaultdict, deque
from config import logger


@dataclass
class PipelineMetrics:
    timestamp: datetime
    ticker_symbol: str
    operation_type: str  # 'fetch', 'compute', 'aggregate', 'forecast'
    records_processed: int
    processing_time: float
    memory_usage: float
    cpu_usage: float
    errors: List[str]
    success: bool
    batch_size: Optional[int] = None
    additional_data: Optional[Dict] = None


@dataclass
class SystemHealth:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    active_threads: int
    db_connections: int


class PipelineMonitor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for global monitoring"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'):
            return

        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 operations
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.system_health_history: deque = deque(maxlen=100)  # Keep last 100 health checks
        self.active_operations: Dict[str, dict] = {}
        self.performance_stats: Dict[str, List[float]] = defaultdict(list)
        self.initialized = True

    def start_operation(self, ticker_symbol: str, operation_type: str = 'general',
                        batch_size: Optional[int] = None) -> str:
        """Start monitoring an operation and return operation ID"""
        operation_id = f"{ticker_symbol}_{operation_type}_{int(time.time())}"

        context = {
            'operation_id': operation_id,
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024,
            'start_cpu': psutil.Process().cpu_percent(),
            'ticker': ticker_symbol,
            'operation_type': operation_type,
            'batch_size': batch_size
        }

        self.active_operations[operation_id] = context

        logger.info(f"🔍 Started monitoring {operation_type} for {ticker_symbol} (ID: {operation_id})")
        return operation_id

    def end_operation(self, operation_id: str, records_processed: int = 0,
                      errors: List[str] = None, success: bool = True,
                      additional_data: Optional[Dict] = None) -> PipelineMetrics:
        """End monitoring and record metrics"""
        if operation_id not in self.active_operations:
            logger.warning(f"Operation ID {operation_id} not found in active operations")
            return None

        context = self.active_operations.pop(operation_id)
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.Process().cpu_percent()

        processing_time = end_time - context['start_time']
        memory_delta = end_memory - context['start_memory']
        cpu_delta = end_cpu - context['start_cpu']

        metrics = PipelineMetrics(
            timestamp=datetime.now(),
            ticker_symbol=context['ticker'],
            operation_type=context['operation_type'],
            records_processed=records_processed,
            processing_time=processing_time,
            memory_usage=memory_delta,
            cpu_usage=cpu_delta,
            errors=errors or [],
            success=success,
            batch_size=context.get('batch_size'),
            additional_data=additional_data or {}
        )

        self.metrics_history.append(metrics)

        # Track error counts
        for error in (errors or []):
            self.error_counts[error] += 1

        # Track performance stats
        self.performance_stats[f"{context['operation_type']}_time"].append(processing_time)
        self.performance_stats[f"{context['operation_type']}_records"].append(records_processed)

        # Log performance metrics
        status_emoji = "✅" if success else "❌"
        logger.info(
            f"{status_emoji} {context['operation_type'].title()} completed for {context['ticker']}: "
            f"Time: {processing_time:.2f}s, Records: {records_processed}, "
            f"Memory: {memory_delta:+.2f}MB, Success: {success}"
        )

        if errors:
            logger.warning(f"⚠️ Errors during {context['operation_type']}: {errors}")

        return metrics

    def record_system_health(self):
        """Record current system health metrics"""
        try:
            health = SystemHealth(
                timestamp=datetime.now(),
                cpu_percent=psutil.cpu_percent(interval=1),
                memory_percent=psutil.virtual_memory().percent,
                disk_usage=psutil.disk_usage('/').percent,
                active_threads=threading.active_count(),
                db_connections=len(self.active_operations)  # Proxy for DB activity
            )

            self.system_health_history.append(health)

            # Log warnings for high resource usage
            if health.cpu_percent > 80:
                logger.warning(f"🔥 High CPU usage: {health.cpu_percent:.1f}%")
            if health.memory_percent > 85:
                logger.warning(f"🧠 High memory usage: {health.memory_percent:.1f}%")

        except Exception as e:
            logger.error(f"Error recording system health: {e}")

    def get_health_summary(self, hours: int = 24) -> dict:
        """Get comprehensive pipeline health summary"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {"status": "no_recent_data", "timestamp": datetime.now().isoformat()}

        # Calculate success rates by operation type
        operation_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'avg_time': 0, 'total_records': 0})

        for m in recent_metrics:
            op_type = m.operation_type
            operation_stats[op_type]['total'] += 1
            if m.success:
                operation_stats[op_type]['success'] += 1
            operation_stats[op_type]['avg_time'] += m.processing_time
            operation_stats[op_type]['total_records'] += m.records_processed

        # Calculate averages
        for op_type in operation_stats:
            stats = operation_stats[op_type]
            if stats['total'] > 0:
                stats['success_rate'] = stats['success'] / stats['total']
                stats['avg_time'] /= stats['total']
            else:
                stats['success_rate'] = 0

        # Overall health
        total_operations = len(recent_metrics)
        successful_operations = sum(1 for m in recent_metrics if m.success)
        overall_success_rate = successful_operations / total_operations if total_operations > 0 else 0

        # Determine health status
        if overall_success_rate >= 0.95:
            health_status = "healthy"
        elif overall_success_rate >= 0.80:
            health_status = "degraded"
        else:
            health_status = "unhealthy"

        # Get latest system health
        latest_system_health = None
        if self.system_health_history:
            latest_health = self.system_health_history[-1]
            latest_system_health = {
                'cpu_percent': latest_health.cpu_percent,
                'memory_percent': latest_health.memory_percent,
                'disk_usage': latest_health.disk_usage,
                'active_threads': latest_health.active_threads,
                'timestamp': latest_health.timestamp.isoformat()
            }

        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "overall_success_rate": overall_success_rate,
                "total_records_processed": sum(m.records_processed for m in recent_metrics),
                "avg_processing_time": sum(
                    m.processing_time for m in recent_metrics) / total_operations if total_operations > 0 else 0
            },
            "operation_breakdown": dict(operation_stats),
            "top_errors": dict(list(sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True))[:10]),
            "active_operations": len(self.active_operations),
            "system_health": latest_system_health
        }

    def get_ticker_performance(self, ticker_symbol: str, hours: int = 24) -> dict:
        """Get performance metrics for a specific ticker"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        ticker_metrics = [m for m in self.metrics_history
                          if m.timestamp > cutoff_time and m.ticker_symbol == ticker_symbol]

        if not ticker_metrics:
            return {"ticker": ticker_symbol, "status": "no_recent_data"}

        total_records = sum(m.records_processed for m in ticker_metrics)
        total_time = sum(m.processing_time for m in ticker_metrics)
        successful_ops = sum(1 for m in ticker_metrics if m.success)

        return {
            "ticker": ticker_symbol,
            "operations_count": len(ticker_metrics),
            "success_rate": successful_ops / len(ticker_metrics),
            "total_records_processed": total_records,
            "total_processing_time": total_time,
            "avg_records_per_second": total_records / total_time if total_time > 0 else 0,
            "last_operation": ticker_metrics[-1].timestamp.isoformat() if ticker_metrics else None
        }

    def export_metrics(self, format: str = 'json', hours: int = 24) -> str:
        """Export metrics in various formats"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]

        if format.lower() == 'json':
            serializable_metrics = []
            for m in recent_metrics:
                metric_dict = asdict(m)
                metric_dict['timestamp'] = m.timestamp.isoformat()
                serializable_metrics.append(metric_dict)

            return json.dumps({
                'export_timestamp': datetime.now().isoformat(),
                'metrics_count': len(serializable_metrics),
                'time_range_hours': hours,
                'metrics': serializable_metrics
            }, indent=2)

        # Add other formats (CSV, etc.) as needed
        return "Unsupported format"

    def cleanup_old_data(self, days: int = 7):
        """Clean up old monitoring data"""
        cutoff_time = datetime.now() - timedelta(days=days)

        # Clean metrics history (handled by deque maxlen, but explicit cleanup)
        initial_count = len(self.metrics_history)
        self.metrics_history = deque(
            [m for m in self.metrics_history if m.timestamp > cutoff_time],
            maxlen=1000
        )

        # Clean system health history
        self.system_health_history = deque(
            [h for h in self.system_health_history if h.timestamp > cutoff_time],
            maxlen=100
        )

        cleaned_count = initial_count - len(self.metrics_history)
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} old monitoring records")


# Global monitor instance
monitor = PipelineMonitor()


# Context manager for easy operation monitoring
class MonitoredOperation:
    def __init__(self, ticker_symbol: str, operation_type: str, batch_size: Optional[int] = None):
        self.ticker_symbol = ticker_symbol
        self.operation_type = operation_type
        self.batch_size = batch_size
        self.operation_id = None
        self.records_processed = 0
        self.errors = []
        self.additional_data = {}

    def __enter__(self):
        self.operation_id = monitor.start_operation(
            self.ticker_symbol, self.operation_type, self.batch_size
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        if exc_type:
            error_msg = f"{exc_type.__name__}: {str(exc_val)}"
            self.errors.append(error_msg)

        monitor.end_operation(
            self.operation_id,
            records_processed=self.records_processed,
            errors=self.errors,
            success=success,
            additional_data=self.additional_data
        )

        return False  # Don't suppress exceptions

    def add_records(self, count: int):
        """Add to the record count"""
        self.records_processed += count

    def add_error(self, error: str):
        """Add an error message"""
        self.errors.append(error)

    def add_data(self, key: str, value: Any):
        """Add additional data"""
        self.additional_data[key] = value