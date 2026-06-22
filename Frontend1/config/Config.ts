

const BACKEND_URL = "http://localhost:9000";
const GET_EMAIL_ENDPOINT = "/gmail/inbox";
const GET_BODY_ENDPOINT = "/gmail/wholemail";

const config = {
    GET_EMAIL_URL: BACKEND_URL + GET_EMAIL_ENDPOINT,
    GET_BODY_URL: BACKEND_URL + GET_BODY_ENDPOINT,
};


export default config;

