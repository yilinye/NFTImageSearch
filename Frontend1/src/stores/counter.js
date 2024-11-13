/*
 * @Description: 
 * @Author: Qing Shi
 * @Date: 2022-09-17 23:36:36
 * @LastEditTime: 2023-01-27 09:26:34
 */
import { fetchHello, postImg, postText, fetchValue } from "../service/module/dataService";
import { ref, computed } from "vue";
import { defineStore } from "pinia";

// export const useCounterStore = defineStore("counter", {
//   const count = ref(0);
//   const doubleCount = computed(() => count.value * 2);
//   function increment() {
//     count.value++;
//   }

//   return { count, doubleCount, increment };
// });

export const useDataStore = defineStore("dataStore", {
  state: () => {
    return {
      msg: 'Hello, Vue SQ',
      imgFile: '',
      imgSet: [],
    }
  },
  actions: {
    fetchHello() {
      const st = new Date();
      fetchHello({}, resp => {
        this.msg = resp;
        console.log("Fetch Hello Time: ", st - new Date());
      })
    },
    postImg({ commit }) {
      const st = new Data();
      postImg({ commit }, resp => {
        // this.imgSet = resp;
        console.log("post Img Time: ", st - new Date());
      })
    },
    postText({ commit }) {
      const st = new Data();
      postText({ commit }, resp => {
        console.log("post Text Time: ", st - new Date());
      })
    },
    fetchValue() {
      const st = new Data();
      fetchValue({}, resp => {
        this.imgSet = resp;
        console.log("Fetch Data Time: ", st - new Date());
      })
    }
  }
})