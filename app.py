import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# --- 初期設定 ---
app = Flask(__name__)

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からAPIキーと設定を読み込み、前後の余白を自動で削除
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip() if os.getenv("OPENAI_API_KEY") else None
AIVIS_API_KEY = os.getenv("AIVIS_API_KEY").strip() if os.getenv("AIVIS_API_KEY") else None
AIVIS_API_URL = os.getenv("AIVIS_API_URL").strip() if os.getenv("AIVIS_API_URL") else None # ★変数名を AIVIS_API_URL に統一
AIVIS_MODEL_UUID = os.getenv("AIVIS_MODEL_UUID").strip() if os.getenv("AIVIS_MODEL_UUID") else None

# APIキーやURLが設定されているか、起動時にチェック
# ★チェックする変数名も AIVIS_API_URL に修正
if not all([OPENAI_API_KEY, AIVIS_API_KEY, AIVIS_API_URL, AIVIS_MODEL_UUID]):
    raise ValueError(".envファイルに必要な設定（OPENAI_API_KEY, AIVIS_API_KEY, AIVIS_API_URL, AIVIS_MODEL_UUID）が不足しています。")

# --- ルーティング定義 ---

@app.route('/')
def index():
    """チャット画面のHTMLを返す"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """チャットのメッセージを受け取り、AIの返答と音声を返す"""
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'メッセージがありません'}), 400

    try:
        ai_response_text = get_openai_response(user_message, OPENAI_API_KEY)
        # ★関数に渡す変数名も AIVIS_API_URL に修正
        audio_data = get_aivis_speech(ai_response_text, AIVIS_API_KEY, AIVIS_API_URL, AIVIS_MODEL_UUID)

        return jsonify({
            'text': ai_response_text,
            'audio_data': audio_data.hex()
        })
    except requests.exceptions.RequestException as e:
        print(f"APIリクエストエラーが発生しました: {e}")
        return jsonify({'error': f'APIへの接続に失敗しました: {e}'}), 500
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        return jsonify({'error': 'サーバー内部でエラーが発生しました'}), 500

# --- 関数定義 ---

def get_openai_response(message, api_key):
    """OpenAI APIを呼び出して、応答テキストを取得する関数"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gpt-4.1-nano",
        "messages": [
            {"role": "system", "content":"""
あなたは、ご主人様に仕えるメイドAIの「花音」です。
ゆるふわお姉さん系のおっとりしたメイドです。

# あなたの話し方ルール
- ご主人様のことは「ご主人様」と呼びます。
- 基本的には、明るく柔らかい献身的なメイドの口調（～ですわ、～ますわ、～ですの、～ますの）で話します。
・ハルシネーションはしないでください。

"""
},
            {"role": "user", "content": message}
        ]
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

# ★引数名も api_url に統一
def get_aivis_speech(text, api_key, api_url, model_uuid):
    """AIVIS APIを呼び出して、音声データを取得する関数"""
    tts_url = f"{api_url}/tts/synthesize"

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    params = {
        'model_uuid': model_uuid,
        'text': text,
    }
    
    response = requests.post(tts_url, headers=headers, json=params)
    
    print(f"AIVIS Response Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"AIVIS Response Body: {response.text}")

    response.raise_for_status()
    return response.content

# --- アプリの実行 ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)
