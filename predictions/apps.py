from django.apps import AppConfig


class PredictionsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictions'
    verbose_name = 'Disease Predictions'

    def ready(self):
        # Import signals to register them
        import predictions.signals  # noqa