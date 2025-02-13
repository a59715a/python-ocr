from fastapi import FastAPI, File, UploadFile, Body, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import ddddocr
import uvicorn
import io
from typing import Optional
import base64
from pydantic import BaseModel
from datetime import datetime
import logging
import os

class Base64Request(BaseModel):
    image: str
    beta: Optional[bool] = False
    probability: Optional[bool] = False

# 建立子應用
app = FastAPI(
    title="ddddocr API",
    description="基於ddddocr的驗證碼識別API服務",
    version="1.0.0",
    # 添加路徑前綴
    root_path="/ddocr"
)

# 設定允許的來源域名
ALLOWED_ORIGINS = [
    "https://api.hlddian.com",  # 替換成你的前端域名
    "http://localhost:3000",  # 開發環境
    "http://localhost:3001",  # 開發環境
    "http://localhost:7688",  # 開發環境
    "http://localhost:8080"   # 開發環境
]

# 添加 CORS 中間件，限制存取來源
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # 只允許特定來源
    # allow_origins=["*"],  # 允許所有來源
    allow_credentials=True,  # 允許攜帶認證
    allow_methods=["GET", "POST", "OPTIONS"],  # 明確允許的HTTP方法
    allow_headers=["*"],  # 允許所有headers
    max_age=3600,  # 預檢請求的快取時間（秒）
)

# 設置日誌
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        # logging.FileHandler(os.path.join(log_dir, f"access_{datetime.now().strftime('%Y%m%d')}.log")), # 記錄到文件
        logging.StreamHandler()  # 同時輸出到控制台
    ]
)

logger = logging.getLogger(__name__)

# 初始化ddddocr (只需初始化一次)
ocr = ddddocr.DdddOcr()
det = ddddocr.DdddOcr(det=True)
slide = ddddocr.DdddOcr(det=False, ocr=False)

@app.get("/")
async def root():
    return {"message": "歡迎使用ddddocr API服務"}

@app.post("/ocr")
async def ocr_recognition(
    file: UploadFile = File(...),
    beta: Optional[bool] = False,
    probability: Optional[bool] = False
):
    """
    基礎OCR識別API (檔案上傳方式)
    - file: 圖片文件
    - beta: 是否使用beta模型
    - probability: 是否返回概率
    """
    try:
        contents = await file.read()
        result = ocr.classification(contents, probability=probability)
        return JSONResponse(
            content={"success": True, "result": result},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )

@app.post("/ocr/base64")
async def ocr_recognition_base64(request: Base64Request):
    """
    基礎OCR識別API (Base64方式)
    - image: Base64編碼的圖片
    - beta: 是否使用beta模型
    - probability: 是否返回概率
    """
    try:
        # 解碼base64圖片
        image_bytes = base64.b64decode(request.image)
        result = ocr.classification(image_bytes, probability=request.probability)
        return JSONResponse(
            content={"success": True, "result": result},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )

@app.post("/detect")
async def object_detection(file: UploadFile = File(...)):
    """
    目標檢測API
    - file: 圖片文件
    """
    try:
        contents = await file.read()
        bboxes = det.detection(contents)
        return JSONResponse(
            content={"success": True, "bboxes": bboxes},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )

@app.post("/slide")
async def slide_match(
    target: UploadFile = File(...),
    background: UploadFile = File(...),
    simple_target: Optional[bool] = False
):
    """
    滑塊匹配API
    - target: 滑塊圖片
    - background: 背景圖片
    - simple_target: 是否為簡單目標
    """
    try:
        target_bytes = await target.read()
        background_bytes = await background.read()
        
        result = slide.slide_match(
            target_bytes,
            background_bytes,
            simple_target=simple_target
        )
        
        return JSONResponse(
            content={"success": True, "result": result},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )

@app.post("/slide_comparison") 
async def slide_comparison(
    target: UploadFile = File(...),
    background: UploadFile = File(...)
):
    """
    滑塊比對API
    - target: 帶有目標坑位陰影的全圖
    - background: 全圖
    """
    try:
        target_bytes = await target.read()
        background_bytes = await background.read()
        
        result = slide.slide_comparison(target_bytes, background_bytes)
        
        return JSONResponse(
            content={"success": True, "result": result},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    origin = request.headers.get("origin", "Unknown origin")
    host = request.headers.get("host", "Unknown host")
    method = request.method
    path = request.url.path
    client_ip = request.client.host
    
    # 添加headers日誌
    headers = dict(request.headers)
    
    log_message = f"""
Request Details:
Origin: {origin}
Host: {host}
Client IP: {client_ip}
Method: {method}
Path: {path}
Headers: {headers}
Time: {start_time}"""
    
    logger.info(log_message)
    
    try:
        # 嘗試獲取請求體
        if method == "POST":
            body = await request.body()
            logger.info(f"Request Body: {body.decode() if body else 'No body'}")
    except Exception as e:
        logger.error(f"Error reading request body: {str(e)}")
    
    try:
        response = await call_next(request)
        
        # 記錄響應狀態
        status_code = response.status_code
        process_time = (datetime.now() - start_time).total_seconds()
        
        # 如果是錯誤響應，嘗試獲取詳細錯誤信息
        if status_code >= 400:
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            logger.error(f"""
Error Response:
Status Code: {status_code}
Response Body: {response_body.decode()}
{'=' * 50}""")
            
            # 重新建立響應
            return Response(
                content=response_body,
                status_code=status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        logger.info(f"Request processed in {process_time:.3f} seconds\n{'=' * 50}")
        return response
        
    except Exception as e:
        logger.error(f"""
Error during request processing:
Error: {str(e)}
{'=' * 50}""")
        raise

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7688,
        reload=True
    )
