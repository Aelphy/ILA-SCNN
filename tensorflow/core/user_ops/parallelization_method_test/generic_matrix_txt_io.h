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

#pragma once

#include <fstream>
#include <stdio.h>

#include <string>
#include <vector>
#include <iostream>
#include <ios>
#include <iomanip>
#include <sstream>
#include <boost/shared_ptr.hpp>
#include <errno.h>


namespace pcc {

    template<typename MatrixT>
    class TXTMatrixReader
    {
    public:
        typedef std::vector< std::vector<MatrixT> > Matrix2d;
        typedef boost::shared_ptr<std::vector<MatrixT> > Vector2DPtr;
        typedef std::vector<Vector2DPtr> Matrix2DPtr;

        TXTMatrixReader(bool a_warn = true) : m_warn(a_warn)
        {}


        inline int
        read (  std::string &a_filename, Matrix2d& m) const
        {
            //open stream
            std::ifstream stream;
            stream.open( a_filename.c_str(), std::ios::out | std::ios::binary );
            if( !stream.is_open() ) {
                std::cerr << "read: can't open file " << a_filename << std::endl;
                return -1;
            }

            //get and write data lines
            std::string line;
            int col_count = 0;
            bool has_warned = false;
            while(std::getline (stream, line, '\n')){
                if(line.size() == 0 || line.at(0) == '#')
                    continue;

                std::vector<MatrixT> elements;
                elements.reserve(col_count);
                int this_col_count = 0;
                std::string elem;
                std::stringstream liness(line);
                while(std::getline (liness, elem, ' ')){
                    if(elem.size() == 0) continue;
                    this_col_count++;
                    std::stringstream valss(elem);
                    MatrixT value(0); valss >> value;
                    elements.push_back(value);
                }
                if(col_count == 0) col_count = this_col_count;
                else if(m_warn && col_count != this_col_count && !has_warned){
                    std::cerr << "matrix is not consistent: the number of columns varries." << std::endl;
                    has_warned = true;
                }
                m.push_back(elements);
            }
            stream.close();
            return 0;
        }

        inline int
        read(std::string &a_filename, Matrix2DPtr& m) const
        {
            //open stream
            std::ifstream stream;
            stream.open( a_filename.c_str(), std::ios::out | std::ios::binary );
            if( !stream.is_open() ) {
                std::cerr << "read: can't open file " << a_filename << std::endl;
                return -1;
            }

            //get and write data lines
            std::string line;
            int col_count = 0;
            bool has_warned = false;
            while(std::getline (stream, line, '\n')){
                if(line.size() == 0 || line.at(0) == '#')
                    continue;

                Vector2DPtr elements(new std::vector<MatrixT>());
                int this_col_count = 0;
                std::string elem;
                std::stringstream liness(line);
                while(std::getline (liness, elem, ' ')){
                    if(elem.size() == 0) continue;
                    this_col_count++;
                    std::stringstream valss(elem);
                    MatrixT value(0); valss >> value;
                    elements->push_back(value);
                }
                if(elements->size() == 0) elements.reset();
                if(col_count == 0) col_count = this_col_count;
                else if(m_warn && col_count != this_col_count && !has_warned){
                    std::cerr << "matrix is not consistent: the number of columns varries." << std::endl;
                    has_warned = true;
                }
                m.push_back(elements);
            }
            stream.close();
            return 0;
        }

        inline int
        read (std::string& a_filename, std::vector<MatrixT> &m, int& a_rows, int& a_cols) const
        {
            //open stream
            std::ifstream stream;
            stream.open( a_filename.c_str(), std::ios::out | std::ios::binary );
            if( !stream.is_open() ) {
                std::cerr << "read: can't open file " << a_filename << std::endl;
                return -1;
            }

            //get and write data lines
            std::string line;
            m.clear();
            a_cols = 0;
            a_rows = 0;
            bool has_warned = false;
            while(std::getline (stream, line, '\n')){
                if(line.size() == 0 || line.at(0) == '#')
                    continue;
                a_rows++;
                int this_col_count = 0;
                std::string elem;
                std::stringstream liness(line);
                while(std::getline (liness, elem, ' ')){
                    if(elem.size() == 0) continue;
                    this_col_count++;
                    std::stringstream valss(elem);
                    MatrixT value(0); valss >> value;
                    m.push_back(value);
                }
                if(a_cols == 0) a_cols = this_col_count;
                else if(m_warn && a_cols != this_col_count && !has_warned){
                    std::cerr << "matrix is not consistent: the number of columns varries." << std::endl;
                    has_warned = true;
                }
            }
            stream.close();
            return 0;
        }
    private:
        bool m_warn;
    };


    template<typename MatrixT>
    class TXTMatrixWriter
    {
    public:
        typedef std::vector< std::vector<MatrixT> > Matrix2d;
        typedef boost::shared_ptr<std::vector<MatrixT> > Vector2DPtr;
        typedef std::vector<Vector2DPtr> Matrix2DPtr;

        TXTMatrixWriter(int a_precision = 15) : m_precision(a_precision)
        {}


        inline int
        write(std::string &a_filename, const Matrix2d &a_matrix, const bool a_append = false) const
        {
            //open stream
            std::ofstream stream;
            if(a_append)
                stream.open( a_filename.c_str(), std::ios::out | std::ios::binary | std::ios::app );
            else
                stream.open( a_filename.c_str(), std::ios::out | std::ios::binary );
            if( !stream.is_open() ) {
                std::cerr << "write: can't open file " << a_filename << std::endl;
                return -1;
            }

            for(size_t i = 0; i < a_matrix.size(); ++i)
            {
                std::stringstream ss;
                for(size_t j = 0; j < a_matrix[i].size(); ++j)
                    ss << std::setprecision(m_precision) << a_matrix[i][j] << " ";
                stream << ss.str() << "\n";
                if(i % 100000 == 0) 
                    stream.flush();
            }
            stream.flush();
            stream.close();
            return 0;
        }

        inline int
        write(std::string &a_filename, const Matrix2DPtr &a_matrix, const bool a_append = false) const
        {
            //open stream
            std::ofstream stream;
            if(a_append)
                stream.open( a_filename.c_str(), std::ios::out | std::ios::binary | std::ios::app );
            else
                stream.open( a_filename.c_str(), std::ios::out | std::ios::binary );
            if( !stream.is_open() ) {
                std::cerr << "write: can't open file " << a_filename << std::endl;
                return -1;
            }

            for(size_t i = 0; i < a_matrix.size(); ++i)
            {
                std::stringstream ss;
                if(a_matrix[i]){
                    for(size_t j = 0; j < a_matrix[i]->size(); ++j)
                        ss << std::setprecision(m_precision) << a_matrix[i]->at(j) << " ";
                }
                stream << ss.str() << "\n";
                if(i % 100000 == 0) 
                    stream.flush();
            }
            stream.flush();
            stream.close();
            return 0;
        }

        inline int
        write(std::string &a_filename, std::vector<MatrixT> &a_matrix, int& a_rows, int& a_cols, const bool a_append = false)
        {
            //open stream
            std::ofstream stream;
            if(a_append)
                stream.open( a_filename.c_str(), std::ios::out | std::ios::binary | std::ios::app );
            else
                stream.open( a_filename.c_str(), std::ios::out | std::ios::binary );
            if( !stream.is_open() ) {
                std::cerr << "write: can't open file " << a_filename << std::endl;
                return -1;
            }

            if(a_rows * a_cols != a_matrix.size()) return -2;

            for(size_t i = 0, k = 0; i < a_rows; ++i)
            {
                std::stringstream ss;
                for(size_t j = 0; j < a_cols; ++j, ++k)
                    ss << std::setprecision(m_precision) << a_matrix[k] << " ";
                stream << ss.str()<< "\n";
                if(i % 100000 == 0) 
                    stream.flush();
            }
            stream.flush();
            stream.close();
            return 0;
        }


    private:
        int m_precision;
    };

    template<typename MatrixT> inline int
    writeTXTFile(std::string a_filename, const std::vector< std::vector<MatrixT> > &a_matrix, const bool a_append = false){
        TXTMatrixWriter<MatrixT> w;
        return w.write(a_filename, a_matrix, a_append);
    }

    template<typename MatrixT> inline int
    loadTXTFile (std::string file_name, std::vector< std::vector<MatrixT> > &a_matrix) {
        TXTMatrixReader<MatrixT> r;
        return r.read(file_name, a_matrix);
    }

    template<typename MatrixT> inline int
    writeTXTFile(std::string a_filename, const std::vector<boost::shared_ptr<std::vector<MatrixT> > > &a_matrix, const bool a_append = false){
        TXTMatrixWriter<MatrixT> w;
        return w.write(a_filename, a_matrix, a_append);
    }

    template<typename MatrixT> inline int
    loadTXTFile (std::string file_name, std::vector<boost::shared_ptr<std::vector<MatrixT> > > &a_matrix) {
        TXTMatrixReader<MatrixT> r;
        return r.read(file_name, a_matrix);
    }


    template<typename MatrixT> inline int
    writeTXTFile(std::string a_filename, std::vector<MatrixT> &a_matrix, int a_rows, int a_cols, const bool a_append = false){
        TXTMatrixWriter<MatrixT> w;
        return w.write(a_filename, a_matrix, a_rows, a_cols, a_append);
    }

    template<typename MatrixT> inline int
    loadTXTFile (std::string file_name, std::vector<MatrixT> &a_matrix, int& a_rows, int& a_cols) {
        TXTMatrixReader<MatrixT> r;
        return r.read(file_name, a_matrix, a_rows, a_cols);
    }
}

