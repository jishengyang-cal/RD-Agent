import { createRouter, RouteRecordRaw, createWebHashHistory } from 'vue-router'

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'Home',
    component: () => import('../views/Home.vue'),
    meta: {
      keepAlive: true, //此页面需要缓存
      footerBg: "#F6FAFF"
    },
  },
  {
    path: '/Playground',
    name: 'Playground',
    component: () => import('../views/Playground.vue'),
    meta: {
      keepAlive: false, //此页面需要缓存
      footerBg: "#fff"
    },
  },
  {
    path: '/PlaygroundPage',
    name: 'PlaygroundPage',
    component: () => import('../views/PlaygroundPage.vue'),
    meta: {
      keepAlive: false, //此页面需要缓存
      footerBg: "#fff"
    },
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router
