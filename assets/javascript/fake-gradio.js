
// Fake gradio components!

// buttons
function newChatClick() {
    gradioApp().querySelector('#empty-btn').click();
}
function jsonDownloadClick() {
    gradioApp().querySelector('#gr-history-download-btn').click();
}
function mdDownloadClick() {
    gradioApp().querySelector('#gr-markdown-export-btn').click();
    gradioApp().querySelector('#gr-history-mardown-download-btn').click();

    // downloadHistory(username, currentChatName, ".md");
}

// index files
function setUploader() {
    transUpload();
    var uploaderObserver = new MutationObserver(function (mutations) {
        var fileInput = null;
        var fileCount = 0;
        fileInput = gradioApp().querySelector("#upload-index-file table.file-preview");
        var fileCountSpan = gradioApp().querySelector("#uploaded-files-count");
        if (fileInput) {
            chatbotArea.classList.add('with-file');
            fileCount = fileInput.querySelectorAll('tbody > tr.file').length;
            fileCountSpan.innerText = fileCount;
        } else {
            chatbotArea.classList.remove('with-file');
            statusDisplayMessage("");
            fileCount = 0;
            transUpload();
        }
    });
    uploaderObserver.observe(uploaderIndicator, {attributes: true})
}
var grUploader;
var chatbotUploader;
var handleClick = function() {
    grUploader.click();

};
function transUpload() {
    chatbotUploader = gradioApp().querySelector("#upload-files-btn");
    chatbotUploader.removeEventListener('click', handleClick);
    grUploader = gradioApp().querySelector("#upload-index-file > .center.flex");

    // let uploaderEvents = ["click", "drag", "dragend", "dragenter", "dragleave", "dragover", "dragstart", "drop"];
    // transEventListeners(chatbotUploader, grUploader, uploaderEvents);

    chatbotUploader.addEventListener('click', handleClick);
}

// checkbox
// var grSingleSessionCB;
// var grOnlineSearchCB;
// var grlightrag_chatCB;
// var chatbotSingleSessionCB;
// var chatbotOnlineSearchCB;
// var grlightrag_chatCB;
// function setCheckboxes() {
//     chatbotSingleSessionCB = gradioApp().querySelector('input[name="single-session-cb"]');
//     chatbotOnlineSearchCB = gradioApp().querySelector('input[name="online-search-cb"]');
//     chatbotlightrag_chatCB = gradioApp().querySelector('input[name="lightrag_chat-cb"]');
//     grSingleSessionCB = gradioApp().querySelector("#gr-single-session-cb > label > input");
//     grOnlineSearchCB = gradioApp().querySelector("#gr-websearch-cb > label> input");
//     grlightrag_chatCB = gradioApp().querySelector("#gr-lightrag_chat-cb > label> input");
    
//     chatbotSingleSessionCB.addEventListener('change', (e) => {
//         grSingleSessionCB.checked = chatbotSingleSessionCB.checked;
//         gradioApp().querySelector('#change-single-session-btn').click();
//     });
//     chatbotOnlineSearchCB.addEventListener('change', (e) => {
//         grOnlineSearchCB.checked = chatbotOnlineSearchCB.checked;
//         gradioApp().querySelector('#change-online-search-btn').click();
//     });
    
//     grSingleSessionCB.addEventListener('change', (e) => {
//         chatbotSingleSessionCB.checked = grSingleSessionCB.checked;
//     });
//     grOnlineSearchCB.addEventListener('change', (e) => {
//         chatbotOnlineSearchCB.checked = grOnlineSearchCB.checked;
//     });
// }

// function bgChangeSingleSession() {
//     // const grSingleSessionCB = gradioApp().querySelector("#gr-single-session-cb > label > input");
//     let a = chatbotSingleSessionCB.checked;
//     return [a];
// }
// function bgChangeOnlineSearch() {
//     // const grOnlineSearchCB = gradioApp().querySelector("#gr-websearch-cb > label> input");
//     let a = chatbotOnlineSearchCB.checked;
//     return [a];
// }

// function updateCheckboxes() {
//     chatbotSingleSessionCB.checked = grSingleSessionCB.checked;
//     chatbotOnlineSearchCB.checked = grOnlineSearchCB.checked;
// }

var grSingleSessionCB;
var grOnlineSearchCB;
var grLightragChatCB; // 修正变量名，避免重复声明
var chatbotSingleSessionCB;
var chatbotOnlineSearchCB;
var chatbotLightragChatCB; // 修正变量名，确保与 Gradio 复选框对应

function setCheckboxes() {
    // 获取聊天机器人的复选框元素
    chatbotSingleSessionCB = gradioApp().querySelector('input[name="single-session-cb"]');
    chatbotOnlineSearchCB = gradioApp().querySelector('input[name="online-search-cb"]');
    chatbotLightragChatCB = gradioApp().querySelector('input[name="lightrag_chat-cb"]');
    
    // 获取 Gradio 复选框元素
    grSingleSessionCB = gradioApp().querySelector("#gr-single-session-cb > label > input");
    grOnlineSearchCB = gradioApp().querySelector("#gr-websearch-cb > label > input");
    grLightragChatCB = gradioApp().querySelector("#gr-lightrag_chat-cb > label > input");
    
    // 监听聊天机器人复选框的变化，并同步到 Gradio 复选框
    chatbotSingleSessionCB.addEventListener('change', (e) => {
        grSingleSessionCB.checked = chatbotSingleSessionCB.checked;
        gradioApp().querySelector('#change-single-session-btn').click();
    });
    
    chatbotOnlineSearchCB.addEventListener('change', (e) => {
        grOnlineSearchCB.checked = chatbotOnlineSearchCB.checked;
        gradioApp().querySelector('#change-online-search-btn').click();
    });
    
    chatbotLightragChatCB.addEventListener('change', (e) => {
        grLightragChatCB.checked = chatbotLightragChatCB.checked;
        gradioApp().querySelector('#change-lightrag_chat-btn').click(); // 确保 Gradio 中有对应的按钮 ID
    });
    
    // 监听 Gradio 复选框的变化，并同步到聊天机器人复选框
    grSingleSessionCB.addEventListener('change', (e) => {
        chatbotSingleSessionCB.checked = grSingleSessionCB.checked;
    });
    
    grOnlineSearchCB.addEventListener('change', (e) => {
        chatbotOnlineSearchCB.checked = grOnlineSearchCB.checked;
    });
    
    grLightragChatCB.addEventListener('change', (e) => {
        chatbotLightragChatCB.checked = grLightragChatCB.checked;
    });
}

function bgChangeSingleSession() {
    let a = chatbotSingleSessionCB.checked;
    return [a];
}

function bgChangeOnlineSearch() {
    let a = chatbotOnlineSearchCB.checked;
    return [a];
}

function bgChangeLightragChat() { // 新增函数以处理 lightrag_chat 的状态变化
    let a = chatbotLightragChatCB.checked;
    return [a];
}

function updateCheckboxes() {
    chatbotSingleSessionCB.checked = grSingleSessionCB.checked;
    chatbotOnlineSearchCB.checked = grOnlineSearchCB.checked;
    chatbotLightragChatCB.checked = grLightragChatCB.checked; // 同步 lightrag_chat 的状态
}

// UTILS
function transEventListeners(target, source, events) {
    events.forEach((sourceEvent) => {
        target.addEventListener(sourceEvent, function (targetEvent) {
            if(targetEvent.preventDefault) targetEvent.preventDefault();
            if(targetEvent.stopPropagation) targetEvent.stopPropagation();

            source.dispatchEvent(new Event(sourceEvent, {detail: targetEvent.detail}));
            // console.log(targetEvent.detail);
        });
    });
}

function bgSelectHistory(a,b){
    const historySelectorInput = gradioApp().querySelector('#history-select-dropdown input');
    let file = historySelectorInput.value;
    return [a,file]
}
