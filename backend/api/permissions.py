from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from django.conf import settings


class APIKeyAuthentication(BaseAuthentication):
    """
    Simple API key authentication using a custom request header.
    """

    class APIUser:
        is_authenticated = True
        pk = 1

    def authenticate(self, request):
        header_name = "HTTP_" + settings.API_KEY_HEADER.replace("-", "_").upper()
        api_key = request.META.get(header_name)

        if not api_key:
            return None  # No API key provided  request is unauthenticated

        if api_key not in settings.VALID_API_KEYS:
            raise AuthenticationFailed("Invalid API key")

        return (self.APIUser(), api_key)
