/**
 * Created by hp on 2018/9/16.
 */



function Nav(){
    var toggle = document.getElementsByClassName('toggle')[0];
    var toggle_nav = document.getElementsByClassName('toggle_nav')[0];
    var boo = true;
    toggle.onclick = function(){
        if(boo){
            toggle_nav.style.height = '710px';
            toggle_nav.style.opacity = '1';
            boo = false;
        }else{
            toggle_nav.style.height = '0';
            toggle_nav.style.opacity = '0';

            boo = true;
        }

    };
}
Nav();