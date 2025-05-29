import unittest
from app import App
import io

class TestApp(unittest.TestCase):

    def setUp(self):
        """Настройка тестового окружения перед каждым тестом."""
        self.app_instance = App()
        self.client = self.app_instance.app.test_client()

    def test_index_route(self):
        """Тестирование главной страницы."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.content_type)

    def test_upload_json_invalid_no_file(self):
        """Тестирование загрузки JSON без файла."""
        response = self.client.post('/upload_json', data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())

    def test_upload_json_invalid_wrong_extension(self):
        """Тестирование загрузки JSON с неправильным расширением файла."""
        data = {
            'demfile': (io.BytesIO(b"dummy data"), 'invalid.txt')
        }
        response = self.client.post('/upload_json', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())

    def test_upload_json_valid(self):
        """Тестирование успешной загрузки JSON с валидным файлом."""
        data = {
            'demfile': (io.BytesIO(b"dummy data"), 'valid.asc')
        }
        response = self.client.post('/upload_json', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('sid', response.get_json())

    def test_stream_session_not_found(self):
        """Тестирование обработки несуществующей сессии."""
        response = self.client.get('/stream/non_existent_sid')
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data.decode(), 'Session not found')

    def test_show_result_not_ready(self):
        """Тестирование обработки запроса результата, когда он не готов."""
        response = self.client.get('/result/non_existent_sid')
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data.decode(), 'Result not ready')


if __name__ == '__main__':
    unittest.main()