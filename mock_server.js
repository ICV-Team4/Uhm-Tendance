const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

// --- 설정 ---
const PORT = 8080;
const IMAGE_PATH = "test.jpg"; 

console.log('Node.js Mock 서버를 시작합니다...');

// 1. 이미지 파일을 읽고 Base64로 미리 인코딩합니다.
let frameBundleJson;
try {
    // 파일 시스템에서 이미지 파일을 Buffer로 읽어옵니다.
    const imageBuffer = fs.readFileSync(IMAGE_PATH);
    
    // Buffer를 "순수한" Base64 텍스트로 변환합니다. (머리말 없음)
    const b64_data = imageBuffer.toString('base64');
    
    // AI 서버가 받을 JSON 페이로드를 생성합니다.
    const frameBundle = {
        version: 1,
        type: "frame_bundle",
        frame_id: 1, // (테스트용이라 1로 고정)
        timestamp: new Date().toISOString(),
        raw_frame: {
            format: "jpeg",
            data: b64_data // "순수" Base64 데이터
        },
        annotated_frame: {
            format: "jpeg",
            data: b64_data // (테스트용이라 원본과 동일하게 보냄)
        },
        boxes: [],
        scores: [],
        names: []
    };
    
    // 나중에 전송할 수 있도록 JSON 문자열로 변환합니다.
    frameBundleJson = JSON.stringify(frameBundle);
    
    console.log(`[Mock Server ${PORT}] 이미지(${IMAGE_PATH}) 로딩 및 Base64 인코딩 성공!`);

} catch (err) {
    console.error(`[ERROR] '${IMAGE_PATH}' 파일을 읽는 중 오류가 발생했습니다!`);
    console.error("Hwa님 dataset 폴더의 실제 이미지 경로로 수정했는지 확인해주세요.");
    console.error(err.message);
    process.exit(1); // 파일이 없으면 서버 실행 중지
}


// 2. 웹소켓 서버를 엽니다.
const wss = new WebSocket.Server({ port: PORT });

wss.on('listening', () => {
    console.log(`[Mock Server ${PORT}] Node.js Mock 서버가 ws://127.0.0.1:${PORT} 에서 실행 중입니다.`);
});

// 3. 클라이언트가 접속하면...
wss.on('connection', ws => {
    console.log(`[Mock Server ${PORT}] 클라이언트(AI서버)가 접속했습니다.`);
    console.log(`[Mock Server ${PORT}] 이미지를 0.5초마다 방송합니다...`);

    // 0.5초마다 미리 준비된 JSON을 방송합니다.
    const interval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
            
            // frame_id와 timestamp를 실시간으로 업데이트
            const liveFrameBundle = JSON.parse(frameBundleJson); // 원본 복사
            liveFrameBundle.frame_id = liveFrameBundle.frame_id + 1;
            liveFrameBundle.timestamp = new Date().toISOString();
            
            ws.send(JSON.stringify(liveFrameBundle));
            
        } else {
            clearInterval(interval);
        }
    }, 500); // 0.5초

    ws.on('close', () => {
        console.log(`[Mock Server ${PORT}] 클라이언트 접속이 끊겼습니다.`);
        clearInterval(interval);
    });
    
    ws.on('error', (err) => {
        console.error(`[Mock Server ${PORT}] 클라이언트 오류:`, err.message);
        clearInterval(interval);
    });
});