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

#ifndef T1H_TIME_H__
#define T1H_TIME_H__


namespace t1h {
	double getTime();
	void sleep(double time);

	//wait for loop rate
	class Rate {
	public:
		Rate(double rate);
 		void wait();

	private:
		double last_time;
		double delta_time;
	};

	//time measurement + advanced timed operations
	class Timer {
	public:
		Timer();

		void 
		tic(); //start timer

		double 
		tac(); //get time since start

		void
		waitForTac(double delta_time); //wait for time from start
	private:
		double last_time;
	};
}

#endif
