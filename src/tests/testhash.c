/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include "hash.h"

const char* words[] = {
    "artichoke","quinoa","dill","kiwi","fingerlime",
    "jabuticaba","lettuce","iceberg","kale","loganberry",
    "date","raspberry","strawberry","honeycrisp","orangelo",
    "endive","quince","tamarind","huckleberry","honeydew",
    "mango","pineapple","apple","miracle","naranjilla",
    "papaya","damson","lemon","gooseberry","cherry",
    "cabbage","yellow","naranjilla","nectarine","zinfandel",
    "zucchini","tangelo","zostera","tamarind","jackfruit",
    "horseradish","yellowhorn","garlic","waxberry","acerola",
    "fennel","orange","tamarind","cherry","watermelon",
    "xigua","the","rambutan","tangerine","garlic",
    "guava","xigua","dragonfruit","by","basil",
    "grape","yellowhorn","kiwi","jabuticaba","quince",
    "cantaloupe","xigua","zucchini","pitanga","parsley",
    "basil","blueberry","jalapeno","zucchini","avocado",
    "fingerlime","miracle","shallot","cherry","voavanga",
    "kiwi","vanilla","raspberry","olive","fig",
    "banana","kiwifruit","jabuticaba","quararibea","mustard",
    "oregano","bilberry","lime","wax","huckleberry",
    "a","dragonfruit","clementine","radicchio","miracle",
    "fingerlime","acerola","huckleberry","pineapple","soursop",
    "thyme","bilberry","elderberry","dill","acerola"
};

int main()
{
    int word_cnt = sizeof(words) / sizeof(words[0]);
    HASHMAP* hmap =  hashmap_create(80,500);
    int inxcnt[word_cnt];
    for (int i = 0; i < word_cnt; i++)
        inxcnt[i] = 0;
    printf("%d words (with duplicates)\n",word_cnt);
    printf("map size %d, initial strings memory size %d\n",hmap->map_size,hmap->mem_size);
    for (int i = 0; i < word_cnt; i++) {
        int inx = hashmap_str2inx(hmap,words[i],1);
        if (inx >= 0) {
            if (inx < word_cnt)
                inxcnt[inx]++;
            else
                printf("Error: at %d: hashmap_str2inx returned unexpected index %d for '%s'\n",i,inx,words[i]);
        }
        else
            printf("Error: at %d: hashmap_str2inx failed for '%s'\n",i,words[i]);
    }
    printf("%d unique words (map size %d)\n",hmap->map_used,hmap->map_size);
    printf("%d bytes of strings memory used out of %d\n",hmap->mem_used,hmap->mem_size);
    for (int i = 0; i < word_cnt; i++) {
        if (inxcnt[i] > 0) {
            const char* word = hashmap_inx2str(hmap,i);
            int inx = hashmap_str2inx(hmap,word,1);
            if (inx != i)
                printf("Error: at %d: hashmap_str2inx returned %d\n",i,inx);
            else            
                printf("%d '%s'\n",inxcnt[i],word);
        }
    }
    hashmap_free(hmap);
    return 0;
}

