/** Copyright (c) 2013, Timo Hackel
*   All rights reserved.
*   
*   Redistribution and use in source and binary forms, with or without
*   modification, are permitted provided that the following conditions are met:
*   
*   1. Redistributions of source code must retain the above copyright notice, this
*      list of conditions and the following disclaimer. 
*   2. Redistributions in binary form must reproduce the above copyright notice,
*      this list of conditions and the following disclaimer in the documentation
*      and/or other materials provided with the distribution.
*   
*   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
*   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
*   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
*   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
*   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
*   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*   
*   The views and conclusions contained in the software and documentation are those
*   of the authors and should not be interpreted as representing official policies, 
*   either expressed or implied, of the FreeBSD Project.
**/

#include "time/time.h"
#include <sys/time.h>
#include <unistd.h>
#include <iostream>

namespace t1h {

    double getTime()
    {
        timeval tv;
        gettimeofday(&tv, NULL);
        return double(tv.tv_sec) + 1./1000000. *  double(tv.tv_usec);
    }

    void sleep(double time)
    {
        if(time > 0)
            usleep(time * 1000000);
    }

    Rate::Rate(double rate) : last_time(0)
    {
        delta_time = 1 / rate;
    }

    void
    Rate::wait()
    {
        if(last_time == 0)
        {
            last_time = t1h::getTime();
            return;
        }
        double new_time = t1h::getTime();

        if(new_time - last_time < delta_time)
        {
            t1h::sleep( new_time - last_time);
        }
        last_time = new_time;
    }

    Timer::Timer() : last_time(0) {}

    void Timer::tic()
    {
        last_time = getTime();
    }

    double Timer::tac()
    {
        return getTime() - last_time;
    }

    void Timer::waitForTac(double delta_time){
        double target = last_time + delta_time;
        double state = getTime();
        sleep(target - state);
    }
}

