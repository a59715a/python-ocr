from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import ddddocr
import uvicorn
import io
from typing import Optional
import base64
from pydantic import BaseModel

class Base64Request(BaseModel):
    image: str
    beta: Optional[bool] = False
    probability: Optional[bool] = False

app = FastAPI(
    title="ddddocr API",
    description="基於ddddocr的驗證碼識別API服務",
    version="1.0.0"
)

# 添加 CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源，生產環境建議設定具體的域名
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有方法
    allow_headers=["*"],  # 允許所有header
)

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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
