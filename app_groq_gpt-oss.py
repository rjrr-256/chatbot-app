import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# --- 初期設定 ---
app = Flask(__name__)
load_dotenv()

# --- 環境変数の読み込み (この部分は元のapp.pyと同じ) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY").strip() if os.getenv("GROQ_API_KEY") else None
AIVIS_API_KEY = os.getenv("AIVIS_API_KEY").strip() if os.getenv("AIVIS_API_KEY") else None
AIVIS_API_URL = os.getenv("AIVIS_API_URL").strip() if os.getenv("AIVIS_API_URL") else None
AIVIS_MODEL_UUID = os.getenv("AIVIS_MODEL_UUID").strip() if os.getenv("AIVIS_MODEL_UUID") else None

if not all([GROQ_API_KEY, AIVIS_API_KEY, AIVIS_API_URL, AIVIS_MODEL_UUID]):
    raise ValueError(".envファイルに必要な設定（GROQ_API_KEY, AIVIS_...）が不足しています。")

# --- ルーティング定義 (この部分は元のapp.pyと同じ) ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'メッセージがありません'}), 400

    try:
        # 呼び出す関数は get_groq_response のまま
        ai_response_text = get_groq_response(user_message, GROQ_API_KEY)
        audio_data = get_aivis_speech(ai_response_text, AIVIS_API_KEY, AIVIS_API_URL, AIVIS_MODEL_UUID)

        return jsonify({
            'text': ai_response_text,
            'audio_data': audio_data.hex()
        })
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return jsonify({'error': 'サーバー内部でエラーが発生しました'}), 500

# --- 関数定義 ---

# ★★★ ここだけを gpt-oss モデル用に変更 ★★★
def get_groq_response(message, api_key):
    """Groq API経由で gpt-oss モデルを呼び出す関数"""
    
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # gpt-oss モデルに合わせたシステムプロンプト
    system_prompt = """
あなたは、ご主人様に仕えるメイドAIの「花音」です。
ゆるふわお姉さん系のおっとりしたメイドです。

# あなたの話し方ルール
- ご主人様のことは「ご主人様」と呼びます。
- 基本的には、明るく柔らかい献身的なメイドの口調（～ですわ、～ますわ、～ですの、～ますの）で話します。
**簡潔に話すこと:** あなたの応答は、常に簡潔で、要点をまとめてください。
・ハルシネーションはしないでください。
"""
    
    data = {
        # ★★★ モデル名を gpt-oss-20b に変更 ★★★
        "model": "openai/gpt-oss-20b",
        
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    
    response = requests.post(api_url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

# 音声合成の関数は変更なし
def get_aivis_speech(text, api_key, api_url, model_uuid):
    tts_url = f"{api_url}/tts/synthesize"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    params = {'model_uuid': model_uuid, 'text': text}
    response = requests.post(tts_url, headers=headers, json=params)
    response.raise_for_status()
    return response.content

# --- アプリの実行 ---
if __name__ == '__main__':
    # ★★★ 実行するポート番号を元のファイルと変えておく ★★★
    # これで、元のアプリと同時に起動できる（もし必要なら）
    app.run(debug=True, port=5002)
