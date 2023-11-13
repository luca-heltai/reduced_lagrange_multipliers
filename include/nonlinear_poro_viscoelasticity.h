/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2010 - 2020 by the deal.II authors and
 *                              Ester Comellas and Jean-Paul Pelteret
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 */

/*  Nonlinear poro-viscoelasticity
 *  Authors: Ester Comellas and Jean-Paul Pelteret,
 *           University of Erlangen-Nuremberg, 2018
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// We start by including all the necessary deal.II header files and some C++
// related ones. They have been discussed in detail in previous tutorial
// programs, so you need only refer to past tutorials for details.

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

// #include <deal.II/base/std_cxx11/shared_ptr.h>
#include <deal.II/base/geometric_utilities.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_linear_operator.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/transformations.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>



// EFI
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#ifndef EFI_CSV_IO_NO_THREAD
#  include <condition_variable>
#  include <mutex>
#  include <thread>

#endif
#include <cassert>
#include <cerrno>
#include <fstream>
#include <istream>
#include <memory>


// For my own writer implementation
#include <boost/filesystem.hpp>

#include <fstream>
#include <iostream>
#include <tuple>



namespace efi
{
  namespace io
  {
    ////////////////////////////////////////////////////////////////////////////
    //                                 LineReader                             //
    ////////////////////////////////////////////////////////////////////////////

    namespace error
    {
      struct base : std::exception
      {
        virtual void
        format_error_message() const = 0;

        const char *
        what() const noexcept override
        {
          format_error_message();
          return error_message_buffer;
        }

        mutable char error_message_buffer[512];
      };

      const int max_file_name_length = 255;

      struct with_file_name
      {
        with_file_name()
        {
          std::memset(file_name, 0, sizeof(file_name));
        }

        void
        set_file_name(const char *file_name)
        {
          if (file_name != nullptr)
            {
              strncpy(this->file_name, file_name, sizeof(this->file_name));
              this->file_name[sizeof(this->file_name) - 1] = '\0';
            }
          else
            {
              this->file_name[0] = '\0';
            }
        }

        char file_name[max_file_name_length + 1];
      };

      struct with_file_line
      {
        with_file_line()
        {
          file_line = -1;
        }

        void
        set_file_line(int file_line)
        {
          this->file_line = file_line;
        }

        int file_line;
      };

      struct with_errno
      {
        with_errno()
        {
          errno_value = 0;
        }

        void
        set_errno(int errno_value)
        {
          this->errno_value = errno_value;
        }

        int errno_value;
      };

      struct can_not_open_file : base, with_file_name, with_errno
      {
        void
        format_error_message() const override
        {
          if (errno_value != 0)
            std::snprintf(error_message_buffer,
                          sizeof(error_message_buffer),
                          "Can not open file \"%s\" because \"%s\".",
                          file_name,
                          std::strerror(errno_value));
          else
            std::snprintf(error_message_buffer,
                          sizeof(error_message_buffer),
                          "Can not open file \"%s\".",
                          file_name);
        }
      };

      struct line_length_limit_exceeded : base, with_file_name, with_file_line
      {
        void
        format_error_message() const override
        {
          std::snprintf(
            error_message_buffer,
            sizeof(error_message_buffer),
            "Line number %d in file \"%s\" exceeds the maximum length of 2^24-1.",
            file_line,
            file_name);
        }
      };
    } // namespace error

    class ByteSourceBase
    {
    public:
      virtual int
      read(char *buffer, int size) = 0;
      virtual ~ByteSourceBase()
      {}
    };

    namespace detail
    {

      class OwningStdIOByteSourceBase : public ByteSourceBase
      {
      public:
        explicit OwningStdIOByteSourceBase(FILE *file)
          : file(file)
        {
          // Tell the std library that we want to do the buffering ourself.
          std::setvbuf(file, 0, _IONBF, 0);
        }

        int
        read(char *buffer, int size) override
        {
          return std::fread(buffer, 1, size, file);
        }

        ~OwningStdIOByteSourceBase()
        {
          std::fclose(file);
        }

      private:
        FILE *file;
      };

      class NonOwningIStreamByteSource : public ByteSourceBase
      {
      public:
        explicit NonOwningIStreamByteSource(std::istream &in)
          : in(in)
        {}

        int
        read(char *buffer, int size) override
        {
          in.read(buffer, size);
          return in.gcount();
        }

        ~NonOwningIStreamByteSource()
        {}

      private:
        std::istream &in;
      };

      class NonOwningStringByteSource : public ByteSourceBase
      {
      public:
        NonOwningStringByteSource(const char *str, long long size)
          : str(str)
          , remaining_byte_count(size)
        {}

        int
        read(char *buffer, int desired_byte_count) override
        {
          int to_copy_byte_count = desired_byte_count;
          if (remaining_byte_count < to_copy_byte_count)
            to_copy_byte_count = remaining_byte_count;
          std::memcpy(buffer, str, to_copy_byte_count);
          remaining_byte_count -= to_copy_byte_count;
          str += to_copy_byte_count;
          return to_copy_byte_count;
        }

        ~NonOwningStringByteSource()
        {}

      private:
        const char *str;
        long long   remaining_byte_count;
      };

#ifndef EFI_CSV_IO_NO_THREAD
      class AsynchronousReader
      {
      public:
        void
        init(std::unique_ptr<ByteSourceBase> arg_byte_source)
        {
          std::unique_lock<std::mutex> guard(lock);
          byte_source           = std::move(arg_byte_source);
          desired_byte_count    = -1;
          termination_requested = false;
          worker                = std::thread([&] {
            std::unique_lock<std::mutex> guard(lock);
            try
              {
                for (;;)
                  {
                    read_requested_condition.wait(guard, [&] {
                      return desired_byte_count != -1 || termination_requested;
                    });
                    if (termination_requested)
                      return;

                    read_byte_count =
                      byte_source->read(buffer, desired_byte_count);
                    desired_byte_count = -1;
                    if (read_byte_count == 0)
                      break;
                    read_finished_condition.notify_one();
                  }
              }
            catch (...)
              {
                read_error = std::current_exception();
              }
            read_finished_condition.notify_one();
          });
        }

        bool
        is_valid() const
        {
          return byte_source != nullptr;
        }

        void
        start_read(char *arg_buffer, int arg_desired_byte_count)
        {
          std::unique_lock<std::mutex> guard(lock);
          buffer             = arg_buffer;
          desired_byte_count = arg_desired_byte_count;
          read_byte_count    = -1;
          read_requested_condition.notify_one();
        }

        int
        finish_read()
        {
          std::unique_lock<std::mutex> guard(lock);
          read_finished_condition.wait(guard, [&] {
            return read_byte_count != -1 || read_error;
          });
          if (read_error)
            std::rethrow_exception(read_error);
          else
            return read_byte_count;
        }

        ~AsynchronousReader()
        {
          if (byte_source != nullptr)
            {
              {
                std::unique_lock<std::mutex> guard(lock);
                termination_requested = true;
              }
              read_requested_condition.notify_one();
              worker.join();
            }
        }

      private:
        std::unique_ptr<ByteSourceBase> byte_source;

        std::thread worker;

        bool               termination_requested;
        std::exception_ptr read_error;
        char              *buffer;
        int                desired_byte_count;
        int                read_byte_count;

        std::mutex              lock;
        std::condition_variable read_finished_condition;
        std::condition_variable read_requested_condition;
      };
#endif

      class SynchronousReader
      {
      public:
        void
        init(std::unique_ptr<ByteSourceBase> arg_byte_source)
        {
          byte_source = std::move(arg_byte_source);
        }

        bool
        is_valid() const
        {
          return byte_source != nullptr;
        }

        void
        start_read(char *arg_buffer, int arg_desired_byte_count)
        {
          buffer             = arg_buffer;
          desired_byte_count = arg_desired_byte_count;
        }

        int
        finish_read()
        {
          return byte_source->read(buffer, desired_byte_count);
        }

      private:
        std::unique_ptr<ByteSourceBase> byte_source;
        char                           *buffer;
        int                             desired_byte_count;
      };
    } // namespace detail

    class LineReader
    {
    private:
      static const int        block_len = 1 << 20;
      std::unique_ptr<char[]> buffer; // must be constructed before (and thus
                                      // destructed after) the reader!
#ifdef EFI_CSV_IO_NO_THREAD
      detail::SynchronousReader reader;
#else
      detail::AsynchronousReader reader;
#endif
      int data_begin;
      int data_end;

      char     file_name[error::max_file_name_length + 1];
      unsigned file_line;

      static std::unique_ptr<ByteSourceBase>
      open_file(const char *file_name)
      {
        // We open the file in binary mode as it makes no difference under *nix
        // and under Windows we handle \r\n newlines ourself.
        FILE *file = std::fopen(file_name, "rb");
        if (file == 0)
          {
            int x = errno; // store errno as soon as possible, doing it after
                           // constructor call can fail.
            error::can_not_open_file err;
            err.set_errno(x);
            err.set_file_name(file_name);
            throw err;
          }
        return std::unique_ptr<ByteSourceBase>(
          new detail::OwningStdIOByteSourceBase(file));
      }

      void
      init(std::unique_ptr<ByteSourceBase> byte_source)
      {
        file_line = 0;

        buffer     = std::unique_ptr<char[]>(new char[3 * block_len]);
        data_begin = 0;
        data_end   = byte_source->read(buffer.get(), 2 * block_len);

        // Ignore UTF-8 BOM
        if (data_end >= 3 && buffer[0] == '\xEF' && buffer[1] == '\xBB' &&
            buffer[2] == '\xBF')
          data_begin = 3;

        if (data_end == 2 * block_len)
          {
            reader.init(std::move(byte_source));
            reader.start_read(buffer.get() + 2 * block_len, block_len);
          }
      }

    public:
      LineReader()                   = delete;
      LineReader(const LineReader &) = delete;
      LineReader &
      operator=(const LineReader &) = delete;

      explicit LineReader(const char *file_name)
      {
        set_file_name(file_name);
        init(open_file(file_name));
      }

      explicit LineReader(const std::string &file_name)
      {
        set_file_name(file_name.c_str());
        init(open_file(file_name.c_str()));
      }

      LineReader(const char                     *file_name,
                 std::unique_ptr<ByteSourceBase> byte_source)
      {
        set_file_name(file_name);
        init(std::move(byte_source));
      }

      LineReader(const std::string              &file_name,
                 std::unique_ptr<ByteSourceBase> byte_source)
      {
        set_file_name(file_name.c_str());
        init(std::move(byte_source));
      }

      LineReader(const char *file_name,
                 const char *data_begin,
                 const char *data_end)
      {
        set_file_name(file_name);
        init(std::unique_ptr<ByteSourceBase>(
          new detail::NonOwningStringByteSource(data_begin,
                                                data_end - data_begin)));
      }

      LineReader(const std::string &file_name,
                 const char        *data_begin,
                 const char        *data_end)
      {
        set_file_name(file_name.c_str());
        init(std::unique_ptr<ByteSourceBase>(
          new detail::NonOwningStringByteSource(data_begin,
                                                data_end - data_begin)));
      }

      LineReader(const char *file_name, FILE *file)
      {
        set_file_name(file_name);
        init(std::unique_ptr<ByteSourceBase>(
          new detail::OwningStdIOByteSourceBase(file)));
      }

      LineReader(const std::string &file_name, FILE *file)
      {
        set_file_name(file_name.c_str());
        init(std::unique_ptr<ByteSourceBase>(
          new detail::OwningStdIOByteSourceBase(file)));
      }

      LineReader(const char *file_name, std::istream &in)
      {
        set_file_name(file_name);
        init(std::unique_ptr<ByteSourceBase>(
          new detail::NonOwningIStreamByteSource(in)));
      }

      LineReader(const std::string &file_name, std::istream &in)
      {
        set_file_name(file_name.c_str());
        init(std::unique_ptr<ByteSourceBase>(
          new detail::NonOwningIStreamByteSource(in)));
      }

      void
      set_file_name(const std::string &file_name)
      {
        set_file_name(file_name.c_str());
      }

      void
      set_file_name(const char *file_name)
      {
        if (file_name != nullptr)
          {
            strncpy(this->file_name, file_name, sizeof(this->file_name));
            this->file_name[sizeof(this->file_name) - 1] = '\0';
          }
        else
          {
            this->file_name[0] = '\0';
          }
      }

      const char *
      get_truncated_file_name() const
      {
        return file_name;
      }

      void
      set_file_line(unsigned file_line)
      {
        this->file_line = file_line;
      }

      unsigned
      get_file_line() const
      {
        return file_line;
      }

      char *
      next_line()
      {
        if (data_begin == data_end)
          return nullptr;

        ++file_line;

        assert(data_begin < data_end);
        assert(data_end <= block_len * 2);

        if (data_begin >= block_len)
          {
            std::memcpy(buffer.get(), buffer.get() + block_len, block_len);
            data_begin -= block_len;
            data_end -= block_len;
            if (reader.is_valid())
              {
                data_end += reader.finish_read();
                std::memcpy(buffer.get() + block_len,
                            buffer.get() + 2 * block_len,
                            block_len);
                reader.start_read(buffer.get() + 2 * block_len, block_len);
              }
          }

        int line_end = data_begin;
        while (buffer[line_end] != '\n' && line_end != data_end)
          {
            ++line_end;
          }

        if (line_end - data_begin + 1 > block_len)
          {
            error::line_length_limit_exceeded err;
            err.set_file_name(file_name);
            err.set_file_line(file_line);
            throw err;
          }

        if (buffer[line_end] == '\n' && line_end != data_end)
          {
            buffer[line_end] = '\0';
          }
        else
          {
            // some files are missing the newline at the end of the
            // last line
            ++data_end;
            buffer[line_end] = '\0';
          }

        // handle windows \r\n-line breaks
        if (line_end != data_begin && buffer[line_end - 1] == '\r')
          buffer[line_end - 1] = '\0';

        char *ret  = buffer.get() + data_begin;
        data_begin = line_end + 1;
        return ret;
      }
    };


    ////////////////////////////////////////////////////////////////////////////
    //                                 CSV                                    //
    ////////////////////////////////////////////////////////////////////////////

    namespace error
    {
      const int max_column_name_length = 63;
      struct with_column_name
      {
        with_column_name()
        {
          std::memset(column_name, 0, max_column_name_length + 1);
        }

        void
        set_column_name(const char *column_name)
        {
          if (column_name != nullptr)
            {
              std::strncpy(this->column_name,
                           column_name,
                           max_column_name_length);
              this->column_name[max_column_name_length] = '\0';
            }
          else
            {
              this->column_name[0] = '\0';
            }
        }

        char column_name[max_column_name_length + 1];
      };


      const int max_column_content_length = 63;

      struct with_column_content
      {
        with_column_content()
        {
          std::memset(column_content, 0, max_column_content_length + 1);
        }

        void
        set_column_content(const char *column_content)
        {
          if (column_content != nullptr)
            {
              std::strncpy(this->column_content,
                           column_content,
                           max_column_content_length);
              this->column_content[max_column_content_length] = '\0';
            }
          else
            {
              this->column_content[0] = '\0';
            }
        }

        char column_content[max_column_content_length + 1];
      };


      struct extra_column_in_header : base, with_file_name, with_column_name
      {
        void
        format_error_message() const override
        {
          std::snprintf(error_message_buffer,
                        sizeof(error_message_buffer),
                        R"(Extra column "%s" in header of file "%s".)",
                        column_name,
                        file_name);
        }
      };

      struct missing_column_in_header : base, with_file_name, with_column_name
      {
        void
        format_error_message() const override
        {
          std::snprintf(error_message_buffer,
                        sizeof(error_message_buffer),
                        R"(Missing column "%s" in header of file "%s".)",
                        column_name,
                        file_name);
        }
      };

      struct duplicated_column_in_header : base,
                                           with_file_name,
                                           with_column_name
      {
        void
        format_error_message() const override
        {
          std::snprintf(error_message_buffer,
                        sizeof(error_message_buffer),
                        R"(Duplicated column "%s" in header of file "%s".)",
                        column_name,
                        file_name);
        }
      };

      struct header_missing : base, with_file_name
      {
        void
        format_error_message() const override
        {
          std::snprintf(error_message_buffer,
                        sizeof(error_message_buffer),
                        "Header missing in file \"%s\".",
                        file_name);
        }
      };

      struct too_few_columns : base, with_file_name, with_file_line
      {
        void
        format_error_message() const override
        {
          std::snprintf(error_message_buffer,
                        sizeof(error_message_buffer),
                        "Too few columns in line %d in file \"%s\".",
                        file_line,
                        file_name);
        }
      };

      struct too_many_columns : base, with_file_name, with_file_line
      {
        void
        format_error_message() const override
        {
          std::snprintf(error_message_buffer,
                        sizeof(error_message_buffer),
                        "Too many columns in line %d in file \"%s\".",
                        file_line,
                        file_name);
        }
      };

      struct escaped_string_not_closed : base, with_file_name, with_file_line
      {
        void
        format_error_message() const override
        {
          std::snprintf(
            error_message_buffer,
            sizeof(error_message_buffer),
            "Escaped string was not closed in line %d in file \"%s\".",
            file_line,
            file_name);
        }
      };

      struct integer_must_be_positive : base,
                                        with_file_name,
                                        with_file_line,
                                        with_column_name,
                                        with_column_content
      {
        void
        format_error_message() const override
        {
          std::snprintf(
            error_message_buffer,
            sizeof(error_message_buffer),
            R"(The integer "%s" must be positive or 0 in column "%s" in file "%s" in line "%d".)",
            column_content,
            column_name,
            file_name,
            file_line);
        }
      };

      struct no_digit : base,
                        with_file_name,
                        with_file_line,
                        with_column_name,
                        with_column_content
      {
        void
        format_error_message() const override
        {
          std::snprintf(
            error_message_buffer,
            sizeof(error_message_buffer),
            R"(The integer "%s" contains an invalid digit in column "%s" in file "%s" in line "%d".)",
            column_content,
            column_name,
            file_name,
            file_line);
        }
      };

      struct integer_overflow : base,
                                with_file_name,
                                with_file_line,
                                with_column_name,
                                with_column_content
      {
        void
        format_error_message() const override
        {
          std::snprintf(
            error_message_buffer,
            sizeof(error_message_buffer),
            R"(The integer "%s" overflows in column "%s" in file "%s" in line "%d".)",
            column_content,
            column_name,
            file_name,
            file_line);
        }
      };

      struct integer_underflow : base,
                                 with_file_name,
                                 with_file_line,
                                 with_column_name,
                                 with_column_content
      {
        void
        format_error_message() const override
        {
          std::snprintf(
            error_message_buffer,
            sizeof(error_message_buffer),
            R"(The integer "%s" underflows in column "%s" in file "%s" in line "%d".)",
            column_content,
            column_name,
            file_name,
            file_line);
        }
      };

      struct invalid_single_character : base,
                                        with_file_name,
                                        with_file_line,
                                        with_column_name,
                                        with_column_content
      {
        void
        format_error_message() const override
        {
          std::snprintf(
            error_message_buffer,
            sizeof(error_message_buffer),
            R"(The content "%s" of column "%s" in file "%s" in line "%d" is not a single character.)",
            column_content,
            column_name,
            file_name,
            file_line);
        }
      };
    } // namespace error

    using ignore_column = unsigned int;
    enum : ignore_column
    {
      ignore_no_column      = 0,
      ignore_extra_column   = 1,
      ignore_missing_column = 2
    };

    template <char... trim_char_list>
    struct trim_chars
    {
    private:
      constexpr static bool
      is_trim_char(char)
      {
        return false;
      }

      template <class... OtherTrimChars>
      constexpr static bool
      is_trim_char(char c, char trim_char, OtherTrimChars... other_trim_chars)
      {
        return c == trim_char || is_trim_char(c, other_trim_chars...);
      }

    public:
      static void
      trim(char *&str_begin, char *&str_end)
      {
        while (str_begin != str_end &&
               is_trim_char(*str_begin, trim_char_list...))
          ++str_begin;
        while (str_begin != str_end &&
               is_trim_char(*(str_end - 1), trim_char_list...))
          --str_end;
        *str_end = '\0';
      }
    };


    struct no_comment
    {
      static bool
      is_comment(const char *)
      {
        return false;
      }
    };

    template <char... comment_start_char_list>
    struct single_line_comment
    {
    private:
      constexpr static bool
      is_comment_start_char(char)
      {
        return false;
      }

      template <class... OtherCommentStartChars>
      constexpr static bool
      is_comment_start_char(char c,
                            char comment_start_char,
                            OtherCommentStartChars... other_comment_start_chars)
      {
        return c == comment_start_char ||
               is_comment_start_char(c, other_comment_start_chars...);
      }

    public:
      static bool
      is_comment(const char *line)
      {
        return is_comment_start_char(*line, comment_start_char_list...);
      }
    };

    struct empty_line_comment
    {
      static bool
      is_comment(const char *line)
      {
        if (*line == '\0')
          return true;
        while (*line == ' ' || *line == '\t')
          {
            ++line;
            if (*line == 0)
              return true;
          }
        return false;
      }
    };

    template <char... comment_start_char_list>
    struct single_and_empty_line_comment
    {
      static bool
      is_comment(const char *line)
      {
        return single_line_comment<comment_start_char_list...>::is_comment(
                 line) ||
               empty_line_comment::is_comment(line);
      }
    };

    template <char sep>
    struct no_quote_escape
    {
      static const char *
      find_next_column_end(const char *col_begin)
      {
        while (*col_begin != sep && *col_begin != '\0')
          ++col_begin;
        return col_begin;
      }

      static void
      unescape(char *&, char *&)
      {}
    };

    template <char sep, char quote>
    struct double_quote_escape
    {
      static const char *
      find_next_column_end(const char *col_begin)
      {
        while (*col_begin != sep && *col_begin != '\0')
          if (*col_begin != quote)
            ++col_begin;
          else
            {
              do
                {
                  ++col_begin;
                  while (*col_begin != quote)
                    {
                      if (*col_begin == '\0')
                        throw error::escaped_string_not_closed();
                      ++col_begin;
                    }
                  ++col_begin;
              } while (*col_begin == quote);
            }
        return col_begin;
      }

      static void
      unescape(char *&col_begin, char *&col_end)
      {
        if (col_end - col_begin >= 2)
          {
            if (*col_begin == quote && *(col_end - 1) == quote)
              {
                ++col_begin;
                --col_end;
                char *out = col_begin;
                for (char *in = col_begin; in != col_end; ++in)
                  {
                    if (*in == quote && (in + 1) != col_end &&
                        *(in + 1) == quote)
                      {
                        ++in;
                      }
                    *out = *in;
                    ++out;
                  }
                col_end  = out;
                *col_end = '\0';
              }
          }
      }
    };

    struct throw_on_overflow
    {
      template <class T>
      static void
      on_overflow(T &)
      {
        throw error::integer_overflow();
      }

      template <class T>
      static void
      on_underflow(T &)
      {
        throw error::integer_underflow();
      }
    };

    struct ignore_overflow
    {
      template <class T>
      static void
      on_overflow(T &)
      {}

      template <class T>
      static void
      on_underflow(T &)
      {}
    };

    struct set_to_max_on_overflow
    {
      template <class T>
      static void
      on_overflow(T &x)
      {
        x = std::numeric_limits<T>::max();
      }

      template <class T>
      static void
      on_underflow(T &x)
      {
        x = std::numeric_limits<T>::min();
      }
    };


    namespace detail
    {
      template <class quote_policy>
      void
      chop_next_column(char *&line, char *&col_begin, char *&col_end)
      {
        assert(line != nullptr);

        col_begin = line;
        // the col_begin + (... - col_begin) removes the constness
        col_end = col_begin +
                  (quote_policy::find_next_column_end(col_begin) - col_begin);

        if (*col_end == '\0')
          {
            line = nullptr;
          }
        else
          {
            *col_end = '\0';
            line     = col_end + 1;
          }
      }

      template <class trim_policy, class quote_policy>
      void
      parse_line(char                   *line,
                 char                  **sorted_col,
                 const std::vector<int> &col_order)
      {
        for (int i : col_order)
          {
            if (line == nullptr)
              throw efi::io::error::too_few_columns();
            char *col_begin, *col_end;
            chop_next_column<quote_policy>(line, col_begin, col_end);

            if (i != -1)
              {
                trim_policy::trim(col_begin, col_end);
                quote_policy::unescape(col_begin, col_end);

                sorted_col[i] = col_begin;
              }
          }
        if (line != nullptr)
          throw efi::io::error::too_many_columns();
      }

      template <unsigned column_count, class trim_policy, class quote_policy>
      void
      parse_header_line(char              *line,
                        std::vector<int>  &col_order,
                        const std::string *col_name,
                        ignore_column      ignore_policy)
      {
        col_order.clear();

        bool found[column_count];
        std::fill(found, found + column_count, false);
        while (line)
          {
            char *col_begin, *col_end;
            chop_next_column<quote_policy>(line, col_begin, col_end);

            trim_policy::trim(col_begin, col_end);
            quote_policy::unescape(col_begin, col_end);

            for (unsigned i = 0; i < column_count; ++i)
              if (col_begin == col_name[i])
                {
                  if (found[i])
                    {
                      error::duplicated_column_in_header err;
                      err.set_column_name(col_begin);
                      throw err;
                    }
                  found[i] = true;
                  col_order.push_back(i);
                  col_begin = 0;
                  break;
                }
            if (col_begin)
              {
                if (ignore_policy & efi::io::ignore_extra_column)
                  col_order.push_back(-1);
                else
                  {
                    error::extra_column_in_header err;
                    err.set_column_name(col_begin);
                    throw err;
                  }
              }
          }
        if (!(ignore_policy & efi::io::ignore_missing_column))
          {
            for (unsigned i = 0; i < column_count; ++i)
              {
                if (!found[i])
                  {
                    error::missing_column_in_header err;
                    err.set_column_name(col_name[i].c_str());
                    throw err;
                  }
              }
          }
      }

      template <class overflow_policy>
      void
      parse(char *col, char &x)
      {
        if (!*col)
          throw error::invalid_single_character();
        x = *col;
        ++col;
        if (*col)
          throw error::invalid_single_character();
      }

      template <class overflow_policy>
      void
      parse(char *col, std::string &x)
      {
        x = col;
      }

      template <class overflow_policy>
      void
      parse(char *col, const char *&x)
      {
        x = col;
      }

      template <class overflow_policy>
      void
      parse(char *col, char *&x)
      {
        x = col;
      }

      template <class overflow_policy, class T>
      void
      parse_unsigned_integer(const char *col, T &x)
      {
        x = 0;
        while (*col != '\0')
          {
            if ('0' <= *col && *col <= '9')
              {
                T y = *col - '0';
                if (x > (std::numeric_limits<T>::max() - y) / 10)
                  {
                    overflow_policy::on_overflow(x);
                    return;
                  }
                x = 10 * x + y;
              }
            else
              throw error::no_digit();
            ++col;
          }
      }

      template <class overflow_policy>
      void
      parse(char *col, unsigned char &x)
      {
        parse_unsigned_integer<overflow_policy>(col, x);
      }
      template <class overflow_policy>
      void
      parse(char *col, unsigned short &x)
      {
        parse_unsigned_integer<overflow_policy>(col, x);
      }
      template <class overflow_policy>
      void
      parse(char *col, unsigned int &x)
      {
        parse_unsigned_integer<overflow_policy>(col, x);
      }
      template <class overflow_policy>
      void
      parse(char *col, unsigned long &x)
      {
        parse_unsigned_integer<overflow_policy>(col, x);
      }
      template <class overflow_policy>
      void
      parse(char *col, unsigned long long &x)
      {
        parse_unsigned_integer<overflow_policy>(col, x);
      }

      template <class overflow_policy, class T>
      void
      parse_signed_integer(const char *col, T &x)
      {
        if (*col == '-')
          {
            ++col;

            x = 0;
            while (*col != '\0')
              {
                if ('0' <= *col && *col <= '9')
                  {
                    T y = *col - '0';
                    if (x < (std::numeric_limits<T>::min() + y) / 10)
                      {
                        overflow_policy::on_underflow(x);
                        return;
                      }
                    x = 10 * x - y;
                  }
                else
                  throw error::no_digit();
                ++col;
              }
            return;
          }
        else if (*col == '+')
          ++col;
        parse_unsigned_integer<overflow_policy>(col, x);
      }

      template <class overflow_policy>
      void
      parse(char *col, signed char &x)
      {
        parse_signed_integer<overflow_policy>(col, x);
      }
      template <class overflow_policy>
      void
      parse(char *col, signed short &x)
      {
        parse_signed_integer<overflow_policy>(col, x);
      }
      template <class overflow_policy>
      void
      parse(char *col, signed int &x)
      {
        parse_signed_integer<overflow_policy>(col, x);
      }
      template <class overflow_policy>
      void
      parse(char *col, signed long &x)
      {
        parse_signed_integer<overflow_policy>(col, x);
      }
      template <class overflow_policy>
      void
      parse(char *col, signed long long &x)
      {
        parse_signed_integer<overflow_policy>(col, x);
      }

      template <class T>
      void
      parse_float(const char *col, T &x)
      {
        bool is_neg = false;
        if (*col == '-')
          {
            is_neg = true;
            ++col;
          }
        else if (*col == '+')
          ++col;

        x = 0;
        while ('0' <= *col && *col <= '9')
          {
            int y = *col - '0';
            x *= 10;
            x += y;
            ++col;
          }

        if (*col == '.' || *col == ',')
          {
            ++col;
            T pos = 1;
            while ('0' <= *col && *col <= '9')
              {
                pos /= 10;
                int y = *col - '0';
                ++col;
                x += y * pos;
              }
          }

        if (*col == 'e' || *col == 'E')
          {
            ++col;
            int e;

            parse_signed_integer<set_to_max_on_overflow>(col, e);

            if (e != 0)
              {
                T base;
                if (e < 0)
                  {
                    base = T(0.1);
                    e    = -e;
                  }
                else
                  {
                    base = T(10);
                  }

                while (e != 1)
                  {
                    if ((e & 1) == 0)
                      {
                        base = base * base;
                        e >>= 1;
                      }
                    else
                      {
                        x *= base;
                        --e;
                      }
                  }
                x *= base;
              }
          }
        else
          {
            if (*col != '\0')
              throw error::no_digit();
          }

        if (is_neg)
          x = -x;
      }

      template <class overflow_policy>
      void
      parse(char *col, float &x)
      {
        parse_float(col, x);
      }
      template <class overflow_policy>
      void
      parse(char *col, double &x)
      {
        parse_float(col, x);
      }
      template <class overflow_policy>
      void
      parse(char *col, long double &x)
      {
        parse_float(col, x);
      }

      template <class overflow_policy, class T>
      void
      parse(char *col, T &x)
      {
        // Mute unused variable compiler warning
        (void)col;
        (void)x;
        // GCC evalutes "false" when reading the template and
        // "sizeof(T)!=sizeof(T)" only when instantiating it. This is why
        // this strange construct is used.
        static_assert(
          sizeof(T) != sizeof(T),
          "Can not parse this type. Only buildin integrals, floats, char, char*, const char* and std::string are supported");
      }

    } // namespace detail

    template <unsigned column_count,
              class trim_policy     = trim_chars<' ', '\t'>,
              class quote_policy    = no_quote_escape<','>,
              class overflow_policy = throw_on_overflow,
              class comment_policy  = no_comment>
    class CSVReader
    {
    private:
      LineReader in;

      char       *row[column_count];
      std::string column_names[column_count];

      std::vector<int> col_order;

      template <class... ColNames>
      void
      set_column_names(std::string s, ColNames... cols)
      {
        column_names[column_count - sizeof...(ColNames) - 1] = std::move(s);
        set_column_names(std::forward<ColNames>(cols)...);
      }

      void
      set_column_names()
      {}


    public:
      CSVReader()                  = delete;
      CSVReader(const CSVReader &) = delete;
      CSVReader &
      operator=(const CSVReader &);

      template <class... Args>
      explicit CSVReader(Args &&...args)
        : in(std::forward<Args>(args)...)
      {
        std::fill(row, row + column_count, nullptr);
        col_order.resize(column_count);
        for (unsigned i = 0; i < column_count; ++i)
          col_order[i] = i;
        for (unsigned i = 1; i <= column_count; ++i)
          column_names[i - 1] = "col" + std::to_string(i);
      }

      char *
      next_line()
      {
        return in.next_line();
      }

      template <class... ColNames>
      void
      read_header(ignore_column ignore_policy, ColNames... cols)
      {
        static_assert(sizeof...(ColNames) >= column_count,
                      "not enough column names specified");
        static_assert(sizeof...(ColNames) <= column_count,
                      "too many column names specified");
        try
          {
            set_column_names(std::forward<ColNames>(cols)...);

            char *line;
            do
              {
                line = in.next_line();
                if (!line)
                  throw error::header_missing();
            } while (comment_policy::is_comment(line));

            detail::parse_header_line<column_count, trim_policy, quote_policy>(
              line, col_order, column_names, ignore_policy);
          }
        catch (error::with_file_name &err)
          {
            err.set_file_name(in.get_truncated_file_name());
            throw;
          }
      }

      template <class... ColNames>
      void
      set_header(ColNames... cols)
      {
        static_assert(sizeof...(ColNames) >= column_count,
                      "not enough column names specified");
        static_assert(sizeof...(ColNames) <= column_count,
                      "too many column names specified");
        set_column_names(std::forward<ColNames>(cols)...);
        std::fill(row, row + column_count, nullptr);
        col_order.resize(column_count);
        for (unsigned i = 0; i < column_count; ++i)
          col_order[i] = i;
      }

      bool
      has_column(const std::string &name) const
      {
        return col_order.end() != std::find(col_order.begin(),
                                            col_order.end(),
                                            std::find(std::begin(column_names),
                                                      std::end(column_names),
                                                      name) -
                                              std::begin(column_names));
      }

      void
      set_file_name(const std::string &file_name)
      {
        in.set_file_name(file_name);
      }

      void
      set_file_name(const char *file_name)
      {
        in.set_file_name(file_name);
      }

      const char *
      get_truncated_file_name() const
      {
        return in.get_truncated_file_name();
      }

      void
      set_file_line(unsigned file_line)
      {
        in.set_file_line(file_line);
      }

      unsigned
      get_file_line() const
      {
        return in.get_file_line();
      }

    private:
      void
      parse_helper(std::size_t)
      {}

      template <class T, class... ColType>
      void
      parse_helper(std::size_t r, T &t, ColType &...cols)
      {
        if (row[r])
          {
            try
              {
                try
                  {
                    efi::io::detail::parse<overflow_policy>(row[r], t);
                  }
                catch (error::with_column_content &err)
                  {
                    err.set_column_content(row[r]);
                    throw;
                  }
              }
            catch (error::with_column_name &err)
              {
                err.set_column_name(column_names[r].c_str());
                throw;
              }
          }
        parse_helper(r + 1, cols...);
      }


    public:
      template <class... ColType>
      bool
      read_row(ColType &...cols)
      {
        static_assert(sizeof...(ColType) >= column_count,
                      "not enough columns specified");
        static_assert(sizeof...(ColType) <= column_count,
                      "too many columns specified");
        try
          {
            try
              {
                char *line;
                do
                  {
                    line = in.next_line();
                    if (!line)
                      return false;
                } while (comment_policy::is_comment(line));

                detail::parse_line<trim_policy, quote_policy>(line,
                                                              row,
                                                              col_order);

                parse_helper(0, cols...);
              }
            catch (error::with_file_name &err)
              {
                err.set_file_name(in.get_truncated_file_name());
                throw;
              }
          }
        catch (error::with_file_line &err)
          {
            err.set_file_line(in.get_file_line());
            throw;
          }
        return true;
      }
    };



    //----------------------------------- Stefan
    //---------------------------------//



    template <unsigned column_count>
    class CSVWriter
    {
    public:
      CSVWriter(const std::string &filename)
        : out(filename, std::ostream::trunc)
      {
        if (!out.is_open())
          {
            // store errno as soon as possible, doing it after constructor
            // call can fail.
            int                      x = errno;
            error::can_not_open_file err;
            err.set_errno(x);
            err.set_file_name(filename.c_str());
            throw err;
          }
      }



      ~CSVWriter()
      {
        if (out.is_open())
          {
            out << std::flush;
            out.close();
          }
      }



      template <class ArrayHead, class... ArrayTail>
      void
      write_rows(const ArrayHead &head, const ArrayTail &...tail)
      {
        static_assert(column_count == (1 + sizeof...(ArrayTail)), " ");
        unsigned int row = 0;
        while (write_row_helper(row, head, tail...))
          {
            ++row;
          }
      }



      template <class Arg, class... Args>
      void
      write_row(const Arg &head, const Args &...tail)
      {
        static_assert(column_count == (1 + sizeof...(Args)), " ");

        std::ostringstream row_out;
        out << head;

        using expander = int[];
        (void)expander{0, (void(out << ", " << tail), 0)...};
        out << "\n";
      }



      template <class ColNamesHead, class... ColNamesTail>
      void
      write_headers(ColNamesHead &&head, ColNamesTail &&...tail)
      {
        static_assert(column_count == (1 + sizeof...(ColNamesTail)), "");

        write_row(std::forward<ColNamesHead>(head),
                  std::forward<ColNamesTail>(tail)...);
      }


    private:
      template <class ArrayType>
      void
      write_element_helper(std::ostream      &element_out,
                           const unsigned int row,
                           const ArrayType   &arr,
                           unsigned int      &count)
      {
        if (row < arr.size())
          {
            ++count;
            element_out << arr[row];
          }
      }


      template <class ArrayHead, class... ArrayTail>
      bool
      write_row_helper(const unsigned int row,
                       const ArrayHead   &head,
                       const ArrayTail &...tail)
      {
        static_assert(column_count == (1 + sizeof...(ArrayTail)), "");

        unsigned int       count = 0;
        std::ostringstream row_out;
        write_element_helper(row_out, row, head, count);

        using expander = int[];
        (void)expander{
          0,
          (void(write_element_helper(row_out << ", ", row, tail, count)),
           0)...};

        if (count > 0)
          {
            out << row_out.str() << "\n";
            return true;
          }

        return false;
      }

      std::ofstream out;
    };



  } // namespace io
} // namespace efi



// We create a namespace for everything that relates to
// the nonlinear poro-viscoelastic formulation,
// and import all the deal.II function and class names into it:
namespace NonLinearPoroViscoElasticity
{
  using namespace dealii;
  using namespace dealii::Functions;

  // @sect3{Run-time parameters}
  //
  // Set up a ParameterHandler object to read in the parameter choices at
  // run-time introduced by the user through the file "parameters.prm"
  namespace Parameters
  {
    // @sect4{Finite Element system}
    // Here we specify the polynomial order used to approximate the solution,
    // both for the displacements and pressure unknowns.
    // The quadrature order should be adjusted accordingly.
    struct Global
    {
      std::string input_directory;

      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    Global::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("global");
      {
        prm.declare_entry("input directory",
                          "",
                          Patterns::FileName(),
                          "input directory path");
      }
      prm.leave_subsection();
    }

    void
    Global::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("global");
      {
        input_directory = prm.get("input directory");
      }
      prm.leave_subsection();
    }


    struct FESystem
    {
      unsigned int poly_degree_displ;
      unsigned int poly_degree_pore;
      unsigned int quad_order;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree displ",
                          "2",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");

        prm.declare_entry("Polynomial degree pore",
                          "1",
                          Patterns::Integer(0),
                          "Pore pressure system polynomial order");

        prm.declare_entry("Quadrature order",
                          "3",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void
    FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree_displ = prm.get_integer("Polynomial degree displ");
        poly_degree_pore  = prm.get_integer("Polynomial degree pore");
        quad_order        = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }

    // @sect4{Geometry}
    // These parameters are related to the geometry definition and mesh
    // generation. We select the type of problem to solve and introduce the
    // desired load values.
    struct Geometry
    {
      std::string  geom_type;
      unsigned int global_refinement;
      double       scale;
      std::string  load_type;
      std::string  input_file;
      double       load;
      unsigned int num_cycle_sets;
      double       fluid_flow;
      double       drained_pressure;
      std::string  lateral_drained;
      std::string  bottom_drained;
      std::string  lateral_confined;
      double       height;
      double       radius;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("geometry@[type=hydro_graz]");
      {
        prm.declare_entry(
          "Geometry type",
          "Ehlers_tube_step_load",
          Patterns::Selection(
            "Ehlers_tube_step_load"
            "|Ehlers_tube_increase_load"
            "|Ehlers_cube_consolidation"
            "|Franceschini_consolidation"
            "|Budday_cube_tension_compression"
            "|Budday_cube_tension_compression_fully_fixed"
            "|Budday_cube_shear_fully_fixed"
            "|Budday_cube_consolidation"
            "|brain_rheometer_shear_lateral_drained"
            "|brain_nanoindentation_sinus"
            "|brain_nanoindentation_ramp"
            "|brain_nanoindentation_flat_ramp"
            "|brain_rheometer_cyclic_trapezoidal_tension_compression"
            "|brain_rheometer_relaxation_tension_compression"
            "|brain_rheometer_cyclic_tension_compression"
            "|brain_rheometer_cyclic_tension_compression_quarter"
            "|brain_rheometer_cyclic_tension_compression_exp"
            "|brain_rheometer_cyclic_tension_compression_exp_quarter"
            "|brain_rheometer_cyclic_compression_quarter"
            "|brain_rheometer_cyclic_tension_quarter"
            "|brain_rheometer_cyclic_trapezoidal_tension_compression_quarter"
            "|brain_rheometer_relaxation_tension_compression_quarter"
            "|brain_rheometer_shear_relaxation_lateral_drained"
            "|hydro_nano_graz_compression_relax"
            "|hydro_nano_graz_compression_exp_relax"
            "|hydro_nano_graz_compression_relax_sphere"
            "|odeometer_graz"),
          "Type of geometry used. "
          "For Ehlers verification examples see Ehlers and Eipper (1999). "
          "For Franceschini brain consolidation see Franceschini et al. (2006)"
          "For Budday brain examples see Budday et al. (2017)");

        prm.declare_entry("Global refinement",
                          "1",
                          Patterns::Integer(0),
                          "Global refinement level");

        prm.declare_entry("Grid scale",
                          "1.0",
                          Patterns::Double(0.0),
                          "Global grid scaling factor");
        prm.declare_entry("height", "8.0", Patterns::Double(0.0));
        prm.declare_entry("radius", "4.0", Patterns::Double(0.0));
      }
      prm.leave_subsection();

      prm.enter_subsection(
        "testing_device@[type=porous_tension_compression_testing_device,instance=1]");
      {
        prm.declare_entry("Load type",
                          "pressure",
                          Patterns::Selection("pressure|displacement|none"),
                          "Type of loading");

        prm.declare_entry("input files",
                          "",
                          Patterns::FileName(),
                          "input file path");

        prm.declare_entry("Load value",
                          "-7.5e+6",
                          Patterns::Double(),
                          "Loading value");

        prm.declare_entry(
          "Number of cycle sets",
          "1",
          Patterns::Integer(1, 3),
          "Number of times each set of 3 cycles is repeated, only for "
          "Budday_cube_tension_compression and Budday_cube_tension_compression_fully_fixed. "
          "Load value is doubled in second set, load rate is kept constant."
          "Final time indicates end of second cycle set.");

        prm.declare_entry(
          "Fluid flow value",
          "0.0",
          Patterns::Double(),
          "Prescribed fluid flow. Not implemented in any example yet.");

        prm.declare_entry(
          "Drained pressure",
          "0.0",
          Patterns::Double(),
          "Increase of pressure value at drained boundary w.r.t the atmospheric pressure.");

        prm.declare_entry("Lateral drained",
                          "drained",
                          Patterns::Selection("drained|undrained"),
                          "Set lateral surfaces to be drained or undrained.");

        prm.declare_entry("Bottom drained",
                          "undrained",
                          Patterns::Selection("drained|undrained"),
                          "Set bottom surface to be drained or undrained.");

        prm.declare_entry(
          "Lateral confined",
          "unconfined",
          Patterns::Selection("confined|unconfined"),
          "Switch between confined and unconfined compression.");
      }
      prm.leave_subsection();
    }


    void
    Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("geometry@[type=hydro_graz]");
      {
        geom_type         = prm.get("Geometry type");
        global_refinement = prm.get_integer("Global refinement");
        scale             = prm.get_double("Grid scale");
        height            = prm.get_double("height");
        radius            = prm.get_double("radius");
      }
      prm.leave_subsection();
      prm.enter_subsection(
        "testing_device@[type=porous_tension_compression_testing_device,instance=1]");
      {
        load_type        = prm.get("Load type");
        input_file       = prm.get("input files");
        load             = prm.get_double("Load value");
        num_cycle_sets   = prm.get_integer("Number of cycle sets");
        fluid_flow       = prm.get_double("Fluid flow value");
        drained_pressure = prm.get_double("Drained pressure");
        lateral_drained  = prm.get("Lateral drained");
        bottom_drained   = prm.get("Bottom drained");
        lateral_confined = prm.get("Lateral confined");
      }
      prm.leave_subsection();
    }

    // @sect4{Materials}

    // Here we select the type of material for the solid component
    // and define the corresponding material parameters.
    // Then we define he fluid data, including the type of
    // seepage velocity definition to use.
    struct Materials
    {
      std::string                           mat_type;
      double                                lambda;
      double                                mu;
      double                                mu1_infty;
      double                                mu2_infty;
      double                                mu3_infty;
      double                                alpha1_infty;
      double                                alpha2_infty;
      double                                alpha3_infty;
      double                                mu1_mode_1;
      double                                mu2_mode_1;
      double                                mu3_mode_1;
      double                                alpha1_mode_1;
      double                                alpha2_mode_1;
      double                                alpha3_mode_1;
      double                                viscosity_mode_1;
      double                                mu1_mode_2;
      double                                mu2_mode_2;
      double                                mu3_mode_2;
      double                                alpha1_mode_2;
      double                                alpha2_mode_2;
      double                                alpha3_mode_2;
      double                                viscosity_mode_2;
      std::string                           fluid_type;
      double                                solid_vol_frac;
      double                                kappa_darcy;
      double                                init_intrinsic_perm;
      double                                viscosity_FR;
      double                                init_darcy_coef;
      double                                weight_FR;
      bool                                  gravity_term;
      int                                   gravity_direction;
      double                                gravity_value;
      double                                density_FR;
      double                                density_SR;
      enum SymmetricTensorEigenvectorMethod eigen_solver;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    Materials::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        prm.declare_entry(
          "material",
          "Neo-Hooke",
          Patterns::Selection(
            "Neo-Hooke|Neo-Hooke-Ehlers|Neo-Hooke-PS|Ogden|visco-Ogden|visco2-Ogden"),
          "Type of material used in the problem");

        prm.declare_entry(
          "lambda",
          "8.375e6",
          Patterns::Double(0, 1e100),
          "First Lamé parameter for extension function related to compactation point in solid material [Pa].");

        prm.declare_entry("shear modulus",
                          "5.583e6",
                          Patterns::Double(0, 1e100),
                          "shear modulus for Neo-Hooke materials [Pa].");

        prm.declare_entry(
          "eigen solver",
          "QL Implicit Shifts",
          Patterns::Selection("QL Implicit Shifts|Jacobi"),
          "The type of eigen solver to be used for Ogden and visco-Ogden models.");

        prm.declare_entry(
          "mu1",
          "0.0",
          Patterns::Double(),
          "Shear material parameter 'mu1' for Ogden material [Pa].");

        prm.declare_entry(
          "mu2",
          "0.0",
          Patterns::Double(),
          "Shear material parameter 'mu2' for Ogden material [Pa].");

        prm.declare_entry(
          "mu3",
          "0.0",
          Patterns::Double(),
          "Shear material parameter 'mu1' for Ogden material [Pa].");

        prm.declare_entry(
          "alpha1",
          "1.0",
          Patterns::Double(),
          "Stiffness material parameter 'alpha1' for Ogden material [-].");

        prm.declare_entry(
          "alpha2",
          "1.0",
          Patterns::Double(),
          "Stiffness material parameter 'alpha2' for Ogden material [-].");

        prm.declare_entry(
          "alpha3",
          "1.0",
          Patterns::Double(),
          "Stiffness material parameter 'alpha3' for Ogden material [-].");

        prm.declare_entry(
          "mu1_1",
          "0.0",
          Patterns::Double(),
          "Shear material parameter 'mu1' for first viscous mode in Ogden material [Pa].");

        prm.declare_entry(
          "mu2_1",
          "0.0",
          Patterns::Double(),
          "Shear material parameter 'mu2' for first viscous mode in Ogden material [Pa].");

        prm.declare_entry(
          "mu3_1",
          "0.0",
          Patterns::Double(),
          "Shear material parameter 'mu1' for first viscous mode in Ogden material [Pa].");

        prm.declare_entry(
          "alpha1_1",
          "1.0",
          Patterns::Double(),
          "Stiffness material parameter 'alpha1' for first viscous mode in Ogden material [-].");

        prm.declare_entry(
          "alpha2_1",
          "1.0",
          Patterns::Double(),
          "Stiffness material parameter 'alpha2' for first viscous mode in Ogden material [-].");

        prm.declare_entry(
          "alpha3_1",
          "1.0",
          Patterns::Double(),
          "Stiffness material parameter 'alpha3' for first viscous mode in Ogden material [-].");

        prm.declare_entry(
          "viscosity_1",
          "1e-10",
          Patterns::Double(1e-10, 1e100),
          "Deformation-independent viscosity parameter 'eta_1' for first viscous mode in Ogden material [-].");

        prm.declare_entry(
          "mu1_2",
          "0.0",
          Patterns::Double(),
          "Shear material parameter 'mu1' for second viscous mode in Ogden material [Pa].");

        prm.declare_entry(
          "mu2_2",
          "0.0",
          Patterns::Double(),
          "Shear material parameter 'mu2' for second viscous mode in Ogden material [Pa].");

        prm.declare_entry(
          "mu3_2",
          "0.0",
          Patterns::Double(),
          "Shear material parameter 'mu1' for second viscous mode in Ogden material [Pa].");

        prm.declare_entry(
          "alpha1_2",
          "1.0",
          Patterns::Double(),
          "Stiffness material parameter 'alpha1' for second viscous mode in Ogden material [-].");

        prm.declare_entry(
          "alpha2_2",
          "1.0",
          Patterns::Double(),
          "Stiffness material parameter 'alpha2' for second viscous mode in Ogden material [-].");

        prm.declare_entry(
          "alpha3_2",
          "1.0",
          Patterns::Double(),
          "Stiffness material parameter 'alpha3' for second viscous mode in Ogden material [-].");

        prm.declare_entry(
          "viscosity_2",
          "1e-10",
          Patterns::Double(1e-10, 1e100),
          "Deformation-independent viscosity parameter 'eta_2' for second viscous mode in Ogden material [-].");

        prm.declare_entry(
          "seepage definition",
          "Ehlers",
          Patterns::Selection("Markert|Ehlers"),
          "Type of formulation used to define the seepage velocity in the problem. "
          "Choose between Markert formulation of deformation-dependent intrinsic permeability "
          "and Ehlers formulation of deformation-dependent Darcy flow coefficient.");

        prm.declare_entry(
          "initial solid volume fraction",
          "0.67",
          Patterns::Double(0.001, 0.999),
          "Initial porosity (solid volume fraction, 0 < n_0s < 1)");

        prm.declare_entry(
          "kappa",
          "0.0",
          Patterns::Double(0, 100),
          "Deformation-dependency control parameter for specific permeability (kappa >= 0)");

        prm.declare_entry(
          "initial intrinsic permeability",
          "0.0",
          Patterns::Double(0, 1e100),
          "Initial intrinsic permeability parameter [m^2] (isotropic permeability). To be used with Markert formulation.");

        prm.declare_entry(
          "fluid viscosity",
          "0.0",
          Patterns::Double(0, 1e100),
          "Effective shear viscosity parameter of the fluid [Pa·s, (N·s)/m^2]. To be used with Markert formulation.");

        prm.declare_entry(
          "initial Darcy coefficient",
          "1.0e-4",
          Patterns::Double(0, 1e100),
          "Initial Darcy flow coefficient [m/s] (isotropic permeability). To be used with Ehlers formulation.");

        prm.declare_entry(
          "fluid weight",
          "1.0e4",
          Patterns::Double(0, 1e100),
          "Effective weight of the fluid [N/m^3]. To be used with Ehlers formulation.");

        prm.declare_entry(
          "gravity term",
          "false",
          Patterns::Bool(),
          "Gravity term considered (true) or neglected (false)");

        prm.declare_entry("fluid density",
                          "1.0",
                          Patterns::Double(0, 1e100),
                          "Real (or effective) density of the fluid");

        prm.declare_entry("solid density",
                          "1.0",
                          Patterns::Double(0, 1e100),
                          "Real (or effective) density of the solid");

        prm.declare_entry(
          "gravity direction",
          "2",
          Patterns::Integer(0, 2),
          "Direction of gravity (unit vector 0 for x, 1 for y, 2 for z)");

        prm.declare_entry(
          "gravity value",
          "-9.81",
          Patterns::Double(),
          "Value of gravity (be careful to have consistent units!)");
      }
      prm.leave_subsection();
    }

    void
    Materials::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        // Solid
        mat_type         = prm.get("material");
        lambda           = prm.get_double("lambda");
        mu               = prm.get_double("shear modulus");
        mu1_infty        = prm.get_double("mu1");
        mu2_infty        = prm.get_double("mu2");
        mu3_infty        = prm.get_double("mu3");
        alpha1_infty     = prm.get_double("alpha1");
        alpha2_infty     = prm.get_double("alpha2");
        alpha3_infty     = prm.get_double("alpha3");
        mu1_mode_1       = prm.get_double("mu1_1");
        mu2_mode_1       = prm.get_double("mu2_1");
        mu3_mode_1       = prm.get_double("mu3_1");
        alpha1_mode_1    = prm.get_double("alpha1_1");
        alpha2_mode_1    = prm.get_double("alpha2_1");
        alpha3_mode_1    = prm.get_double("alpha3_1");
        viscosity_mode_1 = prm.get_double("viscosity_1");
        mu1_mode_2       = prm.get_double("mu1_2");
        mu2_mode_2       = prm.get_double("mu2_2");
        mu3_mode_2       = prm.get_double("mu3_2");
        alpha1_mode_2    = prm.get_double("alpha1_2");
        alpha2_mode_2    = prm.get_double("alpha2_2");
        alpha3_mode_2    = prm.get_double("alpha3_2");
        viscosity_mode_2 = prm.get_double("viscosity_2");
        // Fluid
        fluid_type          = prm.get("seepage definition");
        solid_vol_frac      = prm.get_double("initial solid volume fraction");
        kappa_darcy         = prm.get_double("kappa");
        init_intrinsic_perm = prm.get_double("initial intrinsic permeability");
        viscosity_FR        = prm.get_double("fluid viscosity");
        init_darcy_coef     = prm.get_double("initial Darcy coefficient");
        weight_FR           = prm.get_double("fluid weight");
        // Gravity effects
        gravity_term      = prm.get_bool("gravity term");
        density_FR        = prm.get_double("fluid density");
        density_SR        = prm.get_double("solid density");
        gravity_direction = prm.get_integer("gravity direction");
        gravity_value     = prm.get_double("gravity value");

        if ((fluid_type == "Markert") &&
            ((init_intrinsic_perm == 0.0) || (viscosity_FR == 0.0)))
          AssertThrow(
            false,
            ExcMessage(
              "Markert seepage velocity formulation requires the definition of "
              "'initial intrinsic permeability' and 'fluid viscosity' greater than 0.0."));

        if ((fluid_type == "Ehlers") &&
            ((init_darcy_coef == 0.0) || (weight_FR == 0.0)))
          AssertThrow(
            false,
            ExcMessage(
              "Ehler seepage velocity formulation requires the definition of "
              "'initial Darcy coefficient' and 'fluid weight' greater than 0.0."));

        const std::string eigen_solver_type = prm.get("eigen solver");
        if (eigen_solver_type == "QL Implicit Shifts")
          eigen_solver = SymmetricTensorEigenvectorMethod::ql_implicit_shifts;
        else if (eigen_solver_type == "Jacobi")
          eigen_solver = SymmetricTensorEigenvectorMethod::jacobi;
        else
          {
            AssertThrow(false, ExcMessage("Unknown eigen solver selected."));
          }
      }
      prm.leave_subsection();
    }

    // @sect4{Nonlinear solver}

    // We now define the tolerances and the maximum number of iterations for the
    // Newton-Raphson scheme used to solve the nonlinear system of governing
    // equations.
    struct NonlinearSolver
    {
      unsigned int max_iterations_NR;
      double       tol_f;
      double       tol_u;
      double       tol_p_fluid;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max iterations Newton-Raphson",
                          "15",
                          Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");

        prm.declare_entry("Tolerance force",
                          "1.0e-8",
                          Patterns::Double(0.0),
                          "Force residual tolerance");

        prm.declare_entry("Tolerance displacement",
                          "1.0e-6",
                          Patterns::Double(0.0),
                          "Displacement error tolerance");

        prm.declare_entry("Tolerance pore pressure",
                          "1.0e-6",
                          Patterns::Double(0.0),
                          "Pore pressure error tolerance");
      }
      prm.leave_subsection();
    }

    void
    NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
        tol_f             = prm.get_double("Tolerance force");
        tol_u             = prm.get_double("Tolerance displacement");
        tol_p_fluid       = prm.get_double("Tolerance pore pressure");
      }
      prm.leave_subsection();
    }

    // @sect4{Time}
    // Here we set the timestep size $ \varDelta t $ and the simulation
    // end-time->
    struct Time
    {
      double end_time;
      double end_load_time;
      double delta_t;
      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "10.0", Patterns::Double(), "End time");

        prm.declare_entry("End load time",
                          "-1.0",
                          Patterns::Double(),
                          "End load time");

        prm.declare_entry(
          "Time step size",
          "0.002",
          Patterns::Double(1.0e-6),
          "Time step size. The value must be larger than the displacement error tolerance defined.");
      }
      prm.leave_subsection();
    }

    void
    Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        end_time      = prm.get_double("End time");
        end_load_time = prm.get_double("End load time");
        delta_t       = prm.get_double("Time step size");

        if (end_load_time > end_time)
          AssertThrow(false,
                      ExcMessage(
                        "End load time cannot be larger than End time->"));
      }
      prm.leave_subsection();
    }


    // @sect4{Output}
    // We can choose the frequency of the data for the output files.
    struct OutputParam
    {
      std::string  output_directory;
      std::string  outfiles_requested;
      unsigned int timestep_output;
      std::string  outtype;
      bool         eigenvalue_analysis;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    OutputParam::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Output parameters");
      {
        prm.declare_entry(
          "output directory",
          ".",
          Patterns::FileName(),
          "directory to which the output files will be written");

        prm.declare_entry("Output files",
                          "all",
                          Patterns::Selection("none|bcs|solution|all"),
                          "Paraview output files to generate.");
        prm.declare_entry("Time step number output",
                          "1",
                          Patterns::Integer(0),
                          "Output data for time steps multiple of the given "
                          "integer value.");
        prm.declare_entry("Averaged results",
                          "nodes",
                          Patterns::Selection("elements|nodes"),
                          "Output data associated with integration point values"
                          " averaged on elements or on nodes.");
        prm.declare_entry("Eigenvalue analysis",
                          "false",
                          Patterns::Bool(),
                          "Write output files for eigenvalue analysis");
      }
      prm.leave_subsection();
    }

    void
    OutputParam::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Output parameters");
      {
        output_directory    = prm.get("output directory");
        outfiles_requested  = prm.get("Output files");
        timestep_output     = prm.get_integer("Time step number output");
        outtype             = prm.get("Averaged results");
        eigenvalue_analysis = prm.get_bool("Eigenvalue analysis");
      }
      prm.leave_subsection();
    }

    // @sect4{All parameters}
    // We finally consolidate all of the above structures into a single
    // container that holds all the run-time selections.
    struct AllParameters : public Global,
                           public FESystem,
                           public Geometry,
                           public Materials,
                           public NonlinearSolver,
                           public Time,
                           public OutputParam
    {
      AllParameters(const std::string &input_file);

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    void
    AllParameters::declare_parameters(ParameterHandler &prm)
    {
      Global::declare_parameters(prm);
      FESystem::declare_parameters(prm);
      Geometry::declare_parameters(prm);
      Materials::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      Time::declare_parameters(prm);
      OutputParam::declare_parameters(prm);
    }

    void
    AllParameters::parse_parameters(ParameterHandler &prm)
    {
      Global::parse_parameters(prm);
      FESystem::parse_parameters(prm);
      Geometry::parse_parameters(prm);
      Materials::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      Time::parse_parameters(prm);
      OutputParam::parse_parameters(prm);
    }
  } // namespace Parameters

  // @sect3{Time class}
  // A simple class to store time data.
  // For simplicity we assume a constant time step size.
  class Time
  {
  public:
    Time() = default;

    virtual double
    get_current() const = 0;

    virtual double
    get_end() const = 0;

    virtual double
    get_delta_t() const = 0;

    virtual unsigned int
    get_timestep() const = 0;

    virtual void
    increment_time(double det_F_min = 1) = 0;

    virtual ~Time() = default;
  };

  class TimeFile : public Time
  {
  public:
    TimeFile(const std::string &filename)
      : timestep(0)
      ,
      // delta_t(0.0),
      time_points(0)
    {
      this->read_testfile(filename);
    }

    virtual ~TimeFile()
    {}

    double
    get_current() const override
    {
      if (this->timestep == this->time_points.size())
        return (this->time_points[this->timestep - 1] + 1);
      else
        return this->time_points[this->timestep];
    }
    double
    get_end() const override
    {
      return this->time_points.back();
    }
    double
    get_delta_t() const override
    {
      // Assert ((this->timestep +1) < this->time_points.size(),
      //          ExcMessage("timestep greater then timesteps vector length
      //          -1"))
      double delta_t = 0.;
      // if ((this->timestep + 1) >= this->time_points.size()){
      //   delta_t = this->time_points[this->timestep] -
      //   this->time_points[this->timestep-1];
      //   }
      // else {
      // delta_t = this->time_points[this->timestep+1] -
      // this->time_points[this->timestep];
      delta_t = this->time_points[this->timestep] -
                this->time_points[this->timestep - 1];
      //}
      return delta_t;
    }
    unsigned int
    get_timestep() const override
    {
      return this->timestep;
    }
    void
    increment_time(double det_F_min) override
    {
      (void)det_F_min;
      Assert(this->timestep < this->time_points.size(),
             ExcMessage("timestep exceeds vector length")) this->timestep++;
    }


  private:
    void
    read_testfile(const std::string &filename)
    {
      efi::io::CSVReader<1> in(filename);
      in.read_header(efi::io::ignore_extra_column, "time");

      double time;
      while (in.read_row(time))
        {
          if (this->time_points.empty() != true)
            {
              // Assert (time >= this->time_points.back(),
              //     ExcMessage("decreasing time value found"))
            }
          this->time_points.push_back(time);
        }
    }

    unsigned int timestep;
    // double delta_t;
    std::vector<double> time_points;
  };

  class TimeFixed : public Time
  {
  public:
    TimeFixed(const double time_end, const double time_end_load, double delta_t)
      : dt(delta_t)
      , dt_old(delta_t)
      , cycle_time(1.0)
      , timestep(0)
      , time_current(0.0)
      , time_end(time_end)
      , time_end_load(time_end_load)
      , delta_t(delta_t)
    {}

    virtual ~TimeFixed()
    {}

    double dt;
    double dt_old;
    double cycle_time;

    double
    get_current() const override
    {
      return time_current;
    }
    double
    get_end() const override
    {
      return time_end;
    }
    double
    get_delta_t() const override
    {
      /*double dt    = delta_t;
      double n_0S  = 0.8;
      double n     = 4;
      double range = 0.1;
      double a     = 1/(std::pow(range,n))*delta_t;
      if (time_end_load > 0) {
        if (time_current <= time_end_load && det_F_min > 0.9)
          dt = 1*delta_t;
        else if (time_current <= time_end_load && det_F_min <= 0.9)
          dt = a*std::pow((det_F_min-n_0S),n);
        else if (time_current <= delta_t)
          dt = delta_t - time_current;
        else if (time_current > time_end_load && det_F_min <= 0.9)
          dt = a*std::pow((det_F_min-n_0S),n);
          //dt = delta_t;
      }
      const double multiplier = std::pow(10.0, 6);
      return std::ceil(dt * multiplier) / multiplier;*/
      return dt;
    }
    unsigned int
    get_timestep() const override
    {
      return timestep;
    }
    void
    increment_time(double det_F_min) override
    {
      // double dt_min = 0.1;
      //  based on change in det(F)
      if (time_end_load > 0)
        {
          if (time_current < time_end_load ||
              std::abs(time_current - time_end_load) < 1e-6)
            {
              if (time_current < delta_t)
                dt = delta_t - time_current;
              else if (det_F_min > 0.02)
                { // && dt > dt_min) {
                  dt     = 0.5 * dt;
                  dt_old = dt;
                  if (time_current + dt > time_end_load)
                    dt = time_end_load - time_current;
                }
              else if (std::abs(time_current - time_end_load) < 1e-6)
                dt = 0.1 * dt_old;
              else if (time_current + dt > time_end_load)
                {
                  dt_old = dt;
                  dt     = time_end_load - time_current;
                }
              else if (det_F_min < 0.005 && dt < delta_t)
                {
                  dt     = 2 * dt;
                  dt_old = dt;
                  if (time_current + dt > time_end_load)
                    dt = time_end_load - time_current;
                }
            }
          else
            {
              if (det_F_min > 0.1)
                dt = 0.5 * dt;
              else if (std::abs(time_current - time_end) < 1e-6)
                dt = delta_t;
              else if (time_current + dt > time_end)
                dt = time_end - time_current;
              else if (det_F_min < 0.05 && dt < time_end / 50)
                dt = 2 * dt;
            }
        }

      // for cyclic loading
      if (time_end_load == 0)
        {
          cycle_time = time_end / 12;
          if (std::fmod(time_current, cycle_time) < 1e-6)
            dt = 0.5 * delta_t;
          else if (det_F_min > 0.05)
            dt = 0.5 * dt;
          if (dt > std::abs(std::fmod(time_current, cycle_time) - cycle_time))
            dt = cycle_time - std::fmod(time_current, cycle_time);
          else if (std::abs(time_current - time_end) < 1e-6)
            dt = delta_t;
          else if (time_current + dt > time_end)
            dt = time_end - time_current;
          else if (det_F_min < 0.01 && dt < delta_t)
            dt = 2 * dt;
          if (dt > std::abs(std::fmod(time_current, cycle_time) - cycle_time))
            dt = cycle_time - std::fmod(time_current, cycle_time);
        }

      const double multiplier = std::pow(10.0, 8);
      dt                      = std::ceil(dt * multiplier) / multiplier;

      time_current += dt;
      ++timestep;
    }

  private:
    unsigned int timestep;
    double       time_current;
    double       time_end;
    double       time_end_load;
    double       delta_t;
  };

  // @sect3{Constitutive equation for the solid component of the biphasic
  // material}

  //@sect4{Base class: generic hyperelastic material}
  // The ``extra" Kirchhoff stress in the solid component is the sum of
  // isochoric and a volumetric part.
  // $\mathbf{\tau} = \mathbf{\tau}_E^{(\bullet)} +
  // \mathbf{\tau}^{\textrm{vol}}$ The deviatoric part changes depending on the
  // type of material model selected: Neo-Hooken hyperelasticity, Ogden
  // hyperelasticiy, or a single-mode finite viscoelasticity based on the Ogden
  // hyperelastic model. In this base class we declare  it as a virtual
  // function, and it will be defined for each model type in the corresponding
  // derived class. We define here the volumetric component, which depends on
  // the extension function $U(J_S)$ selected, and in this case is the same for
  // all models. We use the function proposed by Ehlers & Eipper 1999
  // doi:10.1023/A:1006565509095 We also define some public functions to access
  // and update the internal variables.
  template <int dim, typename NumberType = Sacado::Fad::DFad<double>>
  class Material_Hyperelastic
  {
  public:
    Material_Hyperelastic(const Parameters::AllParameters &parameters,
                          const std::shared_ptr<Time>      time)
      : n_OS(parameters.solid_vol_frac)
      , lambda(parameters.lambda)
      , time(time)
      , det_F(1.0)
      , det_F_converged(1.0)
      , eigen_solver(parameters.eigen_solver)
    {}
    virtual ~Material_Hyperelastic()
    {}

    SymmetricTensor<2, dim, NumberType>
    get_tau_E(const Tensor<2, dim, NumberType> &F) const
    {
      return (get_tau_E_base(F) + get_tau_E_ext_func(F));
    }

    SymmetricTensor<2, dim, NumberType>
    get_Cauchy_E(const Tensor<2, dim, NumberType> &F) const
    {
      const NumberType det_F = determinant(F);
      Assert(det_F > 0, ExcInternalError());
      return get_tau_E(F) * NumberType(1 / det_F);
    }

    SymmetricTensor<2, dim, NumberType>
    get_Cauchy_E_base(const Tensor<2, dim, NumberType> &F) const
    {
      const NumberType det_F = determinant(F);
      Assert(det_F > 0, ExcInternalError());
      return get_tau_E_base(F) * NumberType(1 / det_F);
    }

    SymmetricTensor<2, dim, NumberType>
    get_Cauchy_E_ext_func(const Tensor<2, dim, NumberType> &F) const
    {
      const NumberType det_F = determinant(F);
      Assert(det_F > 0, ExcInternalError());
      return get_tau_E_ext_func(F) * NumberType(1 / det_F);
    }

    double
    get_converged_det_F() const
    {
      return det_F_converged;
    }

    virtual void
    update_end_timestep()
    {
      det_F_converged = det_F;
    }

    virtual void
    update_internal_equilibrium(const Tensor<2, dim, NumberType> &F)
    {
      det_F = Tensor<0, dim, double>(determinant(F));
    }

    virtual double
    get_viscous_dissipation() const = 0;

    const double                                n_OS;
    const double                                lambda;
    std::shared_ptr<Time>                       time;
    double                                      det_F;
    double                                      det_F_converged;
    const enum SymmetricTensorEigenvectorMethod eigen_solver;

  protected:
    SymmetricTensor<2, dim, NumberType>
    get_tau_E_ext_func(const Tensor<2, dim, NumberType> &F) const
    {
      const NumberType det_F = determinant(F);
      Assert(det_F > 0, ExcInternalError());
      // Assert(det_F > n_OS, ExcInternalError());

      static const SymmetricTensor<2, dim, double> I(
        Physics::Elasticity::StandardTensors<dim>::I);

      return (NumberType(lambda * (1.0 - n_OS) * (1.0 - n_OS) *
                         (det_F / (1.0 - n_OS) - det_F / (det_F - n_OS))) *
              I);
    }

    virtual SymmetricTensor<2, dim, NumberType>
    get_tau_E_base(const Tensor<2, dim, NumberType> &F) const = 0;
  };

  //@sect4{Derived class: Neo-Hookean hyperelastic material based on a decoupled
  // strain-energy function (see Holzapfel eq. 6.85)}
  template <int dim, typename NumberType = Sacado::Fad::DFad<double>>
  class NeoHooke : public Material_Hyperelastic<dim, NumberType>
  {
  public:
    NeoHooke(const Parameters::AllParameters &parameters,
             const std::shared_ptr<Time>      time)
      : Material_Hyperelastic<dim, NumberType>(parameters, time)
      , mu(parameters.mu)
    {}
    virtual ~NeoHooke()
    {}

    double
    get_viscous_dissipation() const override
    {
      return 0.0;
    }

  protected:
    const double mu;

    SymmetricTensor<2, dim, NumberType>
    get_tau_E_base(const Tensor<2, dim, NumberType> &F) const override
    {
      // left Cauchy-Green strain tensor (see Holzapfel eq. 2.79)
      const SymmetricTensor<2, dim, NumberType> b =
        symmetrize(F * transpose(F));

      // isochoric left Cauchy-Green strain (see Holzapfel eq. 6.99)
      const double det_F = Tensor<0, dim, double>(determinant(F));
      SymmetricTensor<2, dim, NumberType> b_iso = std::pow(det_F, -2.0 / 3) * b;

      // fictitious Kirchhoff stress (see Holzapfel eq. 6.104)
      const SymmetricTensor<2, dim, NumberType> tau_fic = mu * b_iso;

      // isochoric Kirchhoff stress (see Holzapfel eqs. 6.103, 6.105)
      const SymmetricTensor<2, dim, NumberType> tau_iso =
        Physics::Elasticity::StandardTensors<dim>::dev_P * tau_fic;

      return (tau_iso);
    }
  };


  //@sect4{Derived class: Neo-Hookean hyperelastic material based on a decoupled
  // strain-energy function in terms of principal stretches (see Holzapfel
  // eq. 6.85)}
  template <int dim, typename NumberType = Sacado::Fad::DFad<double>>
  class NeoHookePS : public Material_Hyperelastic<dim, NumberType>
  {
  public:
    NeoHookePS(const Parameters::AllParameters &parameters,
               const std::shared_ptr<Time>      time)
      : Material_Hyperelastic<dim, NumberType>(parameters, time)
      , mu(parameters.mu)
    {}
    virtual ~NeoHookePS()
    {}

    double
    get_viscous_dissipation() const override
    {
      return 0.0;
    }

  protected:
    const double mu;

    SymmetricTensor<2, dim, NumberType>
    get_tau_E_base(const Tensor<2, dim, NumberType> &F) const override
    {
      // left Cauchy-Green strain tensor (see Holzapfel eq. 2.79)
      const SymmetricTensor<2, dim, NumberType> b =
        symmetrize(F * transpose(F));

      // isochoric left Cauchy-Green strain in terms of principal stretches
      const double det_F = Tensor<0, dim, double>(determinant(F));
      SymmetricTensor<2, dim, NumberType> b_iso;

      const std::array<std::pair<NumberType, Tensor<1, dim, NumberType>>, dim>
                                 eigen_b = eigenvectors(b, this->eigen_solver);
      Tensor<1, dim, NumberType> lambda_b_iso;

      for (unsigned int a = 0; a < dim; ++a)
        {
          lambda_b_iso[a] = std::pow(det_F, -1.0 / 3) * eigen_b[a].first;

          SymmetricTensor<2, dim, NumberType> ev_basis =
            symmetrize(outer_product(eigen_b[a].second, eigen_b[a].second));
          b_iso += lambda_b_iso[a] * ev_basis;
        }

      // fictitious Kirchhoff stress (see Holzapfel eq. 6.104)
      const SymmetricTensor<2, dim, NumberType> tau_fic = mu * b_iso;

      // isochoric Kirchhoff stress (see Holzapfel eqs. 6.103, 6.105)
      const SymmetricTensor<2, dim, NumberType> tau_iso =
        Physics::Elasticity::StandardTensors<dim>::dev_P * tau_fic;

      return (tau_iso);
    }
  };


  //@sect4{Derived class: Neo-Hookean hyperelastic material as in Ehlers &
  // Eipper (1999), compare eq. 33}
  template <int dim, typename NumberType = Sacado::Fad::DFad<double>>
  class NeoHookeEhlers : public Material_Hyperelastic<dim, NumberType>
  {
  public:
    NeoHookeEhlers(const Parameters::AllParameters &parameters,
                   const std::shared_ptr<Time>      time)
      : Material_Hyperelastic<dim, NumberType>(parameters, time)
      , mu(parameters.mu)
    {}
    virtual ~NeoHookeEhlers()
    {}

    double
    get_viscous_dissipation() const override
    {
      return 0.0;
    }

  protected:
    const double mu;

    SymmetricTensor<2, dim, NumberType>
    get_tau_E_base(const Tensor<2, dim, NumberType> &F) const override
    {
      static const SymmetricTensor<2, dim, double> I(
        Physics::Elasticity::StandardTensors<dim>::I);
      return (mu * (symmetrize(F * transpose(F)) - I));
    }
  };


  //@sect4{Derived class: Ogden hyperelastic material}
  template <int dim, typename NumberType = Sacado::Fad::DFad<double>>
  class Ogden : public Material_Hyperelastic<dim, NumberType>
  {
  public:
    Ogden(const Parameters::AllParameters &parameters,
          const std::shared_ptr<Time>      time)
      : Material_Hyperelastic<dim, NumberType>(parameters, time)
      , mu_infty(
          {parameters.mu1_infty, parameters.mu2_infty, parameters.mu3_infty})
      , alpha_infty({parameters.alpha1_infty,
                     parameters.alpha2_infty,
                     parameters.alpha3_infty})
    {}
    virtual ~Ogden()
    {}

    double
    get_viscous_dissipation() const override
    {
      return 0.0;
    }

  protected:
    std::vector<double> mu_infty;
    std::vector<double> alpha_infty;

    SymmetricTensor<2, dim, NumberType>
    get_tau_E_base(const Tensor<2, dim, NumberType> &F) const override
    {
      // left Cauchy-Green strain tensor (see Holzapfel eq. 2.79)
      const SymmetricTensor<2, dim, NumberType> b =
        symmetrize(F * transpose(F));

      // get squared principal stretches (eigenvalues of b) and eigenvectors of
      // b
      const std::array<std::pair<NumberType, Tensor<1, dim, NumberType>>, dim>
        eigen_b = eigenvectors(b, this->eigen_solver);

      //			const SymmetricTensor< 2, dim, NumberType> I
      //(Physics::Elasticity::StandardTensors<dim>::I); 			std::array<
      // std::pair<
      // NumberType, Tensor< 1, dim, NumberType > >, dim > 			eigen_I =
      // eigenvectors(I, this->eigen_solver); 			eigen_I[0].second[0] = 1;
      // eigen_I[0].second[1] = 0; eigen_I[0].second[2] = 0;
      // eigen_I[1].second[0] = 0; eigen_I[1].second[1] = 1;
      // eigen_I[1].second[2] = 0; 			eigen_I[2].second[0] = 0;
      // eigen_I[2].second[1] = 0; eigen_I[2].second[2] = 1;

      // compute isochoric principal stretches (see Holzapfel eq. 6.81)
      // double                     det_F = Tensor<0, dim,
      // double>(determinant(F));
      Tensor<1, dim, NumberType> lambda_iso;

      for (unsigned int a = 0; a < dim; ++a)
        {
          // lambda_iso[a] = std::pow(det_F, -1.0/3) *
          // std::sqrt(eigen_b[a].first);
          lambda_iso[a] = std::sqrt(eigen_b[a].first);
        }

      // compute Kirchhoff stress tensor (see Comellas (2020) eq. 52)
      SymmetricTensor<2, dim, NumberType> tau_iso;
      for (unsigned int a = 0; a < dim; ++a)
        {
          SymmetricTensor<2, dim, NumberType> ev_basis =
            symmetrize(outer_product(eigen_b[a].second, eigen_b[a].second));
          // SymmetricTensor<2, dim, NumberType> ev_basis_I =
          // symmetrize(outer_product(eigen_I[a].second,eigen_I[a].second));
          //  Print deformation gradient to file
          //	        	std::cout
          //						<< std::setw(16) << ev_basis_I[0][0] << ","
          //						<< std::setw(16) << ev_basis_I[0][1] << ","
          //						<< std::setw(16) << ev_basis_I[0][2] << ","
          //						<< std::setw(16) << ev_basis_I[1][0] << ","
          //						<< std::setw(16) << ev_basis_I[1][1] << ","
          //						<< std::setw(16) << ev_basis_I[1][2] << ","
          //						<< std::setw(16) << ev_basis_I[2][0] << ","
          //						<< std::setw(16) << ev_basis_I[2][1] << ","
          //						<< std::setw(16) << ev_basis_I[2][2] << "," <<
          // std::endl;
          tau_iso += get_beta_infty(lambda_iso, a) * ev_basis;
          // tau_iso += get_beta_infty(lambda_iso, a) * ev_basis_I;
        }



      return tau_iso;
    }

    //		// compare Comellas (2020) eq. 52
    //		NumberType get_beta_infty(Tensor<1, dim, NumberType> &lambda_iso,
    // const unsigned int &a) const
    //		{
    //			NumberType beta = 0.0;
    //
    //			for (unsigned int i = 0; i < 3; ++i) //3rd-order Ogden model
    //			{
    //				NumberType aux = 0.0;
    //				for (int p = 0; p < dim; ++p)
    //					aux += std::pow(lambda_iso[p],alpha_infty[i]);
    //
    //				aux *= -1.0/dim;
    //				aux += std::pow(lambda_iso[a], alpha_infty[i]);
    //				aux *= mu_infty[i];
    //
    //				beta  += aux;
    //			}
    //			return beta;
    //		}

    NumberType
    get_beta_infty(Tensor<1, dim, NumberType> &lambda_iso,
                   const unsigned int         &a) const
    {
      NumberType beta = 0.0;

      for (unsigned int i = 0; i < 3; ++i) // 3rd-order Ogden model
        {
          NumberType aux = 0.0;
          aux            = std::pow(lambda_iso[a], alpha_infty[i]);
          aux -= 1;
          aux *= mu_infty[i];

          beta += aux;
        }
      return beta;
    }
  };

  //@sect4{Derived class: Single-mode Ogden viscoelastic material}
  // We use the finite viscoelastic model described in
  // Reese & Govindjee (1998) doi:10.1016/S0020-7683(97)00217-5
  // The algorithm for the implicit exponential time integration is given in
  // Budday et al. (2017) doi: 10.1016/j.actbio.2017.06.024
  template <int dim, typename NumberType = Sacado::Fad::DFad<double>>
  class visco_Ogden : public Material_Hyperelastic<dim, NumberType>
  {
  public:
    visco_Ogden(const Parameters::AllParameters &parameters,
                const std::shared_ptr<Time>      time)
      : Material_Hyperelastic<dim, NumberType>(parameters, time)
      , mu_infty(
          {parameters.mu1_infty, parameters.mu2_infty, parameters.mu3_infty})
      , alpha_infty({parameters.alpha1_infty,
                     parameters.alpha2_infty,
                     parameters.alpha3_infty})
      , mu_mode_1(
          {parameters.mu1_mode_1, parameters.mu2_mode_1, parameters.mu3_mode_1})
      , alpha_mode_1({parameters.alpha1_mode_1,
                      parameters.alpha2_mode_1,
                      parameters.alpha3_mode_1})
      , viscosity_mode_1(parameters.viscosity_mode_1)
      , Cinv_v_1(Physics::Elasticity::StandardTensors<dim>::I)
      , Cinv_v_1_converged(Physics::Elasticity::StandardTensors<dim>::I)
    {}
    virtual ~visco_Ogden()
    {}

    void
    update_internal_equilibrium(const Tensor<2, dim, NumberType> &F) override
    {
      Material_Hyperelastic<dim, NumberType>::update_internal_equilibrium(F);

      SymmetricTensor<2, dim> F_print;

      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          {
            F_print[i][j] = Tensor<0, dim, double>(F[i][j]);
          }
      // Print cell date to eigenvalue file
      std::ofstream F_out;
      F_out.open("F_out", std::ofstream::app);
      F_out << std::setprecision(6) << std::scientific;
      F_out << std::setw(16) << this->time->get_current() << ","
            << std::setw(16) << F_print[0][0] << "," << std::setw(16)
            << F_print[0][1] << "," << std::setw(16) << F_print[0][2] << ","
            << std::setw(16) << F_print[1][0] << "," << std::setw(16)
            << F_print[1][1] << "," << std::setw(16) << F_print[1][2] << ","
            << std::setw(16) << F_print[2][0] << "," << std::setw(16)
            << F_print[2][1] << "," << std::setw(16) << F_print[2][2] << ","
            << std::endl;
      F_out.close();


      this->Cinv_v_1 = this->Cinv_v_1_converged;
      SymmetricTensor<2, dim, NumberType> B_e_1_tr =
        symmetrize(F * this->Cinv_v_1 * transpose(F));

      const std::array<std::pair<NumberType, Tensor<1, dim, NumberType>>, dim>
        eigen_B_e_1_tr = eigenvectors(B_e_1_tr, this->eigen_solver);

      NumberType J_e_1 = std::sqrt(determinant(B_e_1_tr));

      Tensor<1, dim, NumberType> lambdas_e_1_tr;
      Tensor<1, dim, NumberType> epsilon_e_1_tr;
      Tensor<1, dim, NumberType> lambdas_e_1_iso_tr; // remove
      Tensor<1, dim, NumberType> epsilon_e_1_iso_tr; // remove
      for (int a = 0; a < dim; ++a)
        {
          lambdas_e_1_tr[a] = std::sqrt(eigen_B_e_1_tr[a].first);
          epsilon_e_1_tr[a] = std::log(lambdas_e_1_tr[a]);
          lambdas_e_1_iso_tr[a] =
            lambdas_e_1_tr[a] * std::pow(J_e_1, -1.0 / dim);       // remove
          epsilon_e_1_iso_tr[a] = std::log(lambdas_e_1_iso_tr[a]); // remove
        }

      const double               tolerance      = 1e-8;
      double                     residual_check = tolerance * 10.0;
      Tensor<1, dim, NumberType> residual;
      Tensor<2, dim, NumberType> tangent;
      static const SymmetricTensor<2, dim, double> I(
        Physics::Elasticity::StandardTensors<dim>::I);

      std::vector<NumberType>             lambdas_e_1_iso(dim);
      Tensor<1, dim, NumberType>          epsilon_e_1_iso; // remove
      SymmetricTensor<2, dim, NumberType> B_e_1;
      int                                 iteration = 0;

      Tensor<1, dim, NumberType> lambdas_e_1;
      Tensor<1, dim, NumberType> epsilon_e_1;
      epsilon_e_1     = epsilon_e_1_tr;
      epsilon_e_1_iso = epsilon_e_1_iso_tr;

      while (residual_check > tolerance)
        {
          NumberType aux_J_e_1 = 1.0;
          for (unsigned int a = 0; a < dim; ++a)
            {
              lambdas_e_1[a] = std::exp(epsilon_e_1[a]);
              aux_J_e_1 *= lambdas_e_1[a];
            }

          J_e_1 = aux_J_e_1;
          // double J1 = Tensor<0,dim,double>(J_e_1);
          // std::cout << "J = " << J1 << " in iteration " << iteration <<
          // std::endl;

          for (unsigned int a = 0; a < dim; ++a)
            {
              lambdas_e_1_iso[a] = lambdas_e_1[a] * std::pow(J_e_1, -1.0 / dim);
              // lambdas_e_1_iso[a] = std::exp(epsilon_e_1_iso[a]); //remove
              // epsilon_e_1_iso[a] = std::log(lambdas_e_1_iso[a]); //remove
            }

          for (unsigned int a = 0; a < dim; ++a)
            {
              residual[a] = get_beta_mode_1(lambdas_e_1_iso, a);
              residual[a] *=
                this->time->get_delta_t() / (2.0 * viscosity_mode_1);
              residual[a] += epsilon_e_1[a];
              residual[a] -= epsilon_e_1_tr[a];
              // residual[a] += epsilon_e_1_iso[a]; //remove
              // residual[a] -= epsilon_e_1_iso_tr[a]; //remove

              for (unsigned int b = 0; b < dim; ++b)
                {
                  tangent[a][b] = get_gamma_mode_1(lambdas_e_1_iso, a, b);
                  tangent[a][b] *=
                    this->time->get_delta_t() / (2.0 * viscosity_mode_1);
                  tangent[a][b] += I[a][b];
                }
            }
          epsilon_e_1 -= invert(tangent) * residual;
          // epsilon_e_1_iso -= invert(tangent)*residual; //remove

          residual_check = 0.0;
          for (unsigned int a = 0; a < dim; ++a)
            {
              if (std::abs(residual[a]) > residual_check)
                residual_check = std::abs(Tensor<0, dim, double>(residual[a]));
            }
          iteration += 1;
          if (iteration > 15)
            AssertThrow(
              false,
              ExcMessage(
                "No convergence in local Newton iteration for the "
                "viscoelastic exponential time integration algorithm."));
        }

      NumberType aux_J_e_1 = 1.0;
      for (unsigned int a = 0; a < dim; ++a)
        {
          lambdas_e_1[a] = std::exp(epsilon_e_1[a]);
          aux_J_e_1 *= lambdas_e_1[a];
          // lambdas_e_1_iso[a] = std::exp(epsilon_e_1_iso[a]); //remove
        }
      J_e_1 = aux_J_e_1;

      for (unsigned int a = 0; a < dim; ++a)
        lambdas_e_1_iso[a] = lambdas_e_1[a] * std::pow(J_e_1, -1.0 / dim);
      // lambdas_e_1[a] = lambdas_e_1_iso[a]*std::pow(J_e_1,1.0/dim);

      for (unsigned int a = 0; a < dim; ++a)
        {
          SymmetricTensor<2, dim, NumberType> B_e_1_aux = symmetrize(
            outer_product(eigen_B_e_1_tr[a].second, eigen_B_e_1_tr[a].second));
          B_e_1_aux *= lambdas_e_1[a] * lambdas_e_1[a];
          B_e_1 += B_e_1_aux;
        }

      Tensor<2, dim, NumberType> Cinv_v_1_AD =
        symmetrize(invert(F) * B_e_1 * invert(transpose(F)));

      this->tau_neq_1 = 0;
      for (unsigned int a = 0; a < dim; ++a)
        {
          SymmetricTensor<2, dim, NumberType> tau_neq_1_aux = symmetrize(
            outer_product(eigen_B_e_1_tr[a].second, eigen_B_e_1_tr[a].second));
          tau_neq_1_aux *= get_beta_mode_1(lambdas_e_1_iso, a);
          this->tau_neq_1 += tau_neq_1_aux;
        }

      // Store history
      for (unsigned int a = 0; a < dim; ++a)
        for (unsigned int b = 0; b < dim; ++b)
          this->Cinv_v_1[a][b] = Tensor<0, dim, double>(Cinv_v_1_AD[a][b]);
    }

    void
    update_end_timestep() override
    {
      Material_Hyperelastic<dim, NumberType>::update_end_timestep();
      this->Cinv_v_1_converged = this->Cinv_v_1;
    }

    double
    get_viscous_dissipation() const override
    {
      NumberType dissipation_term =
        get_tau_E_neq() *
        get_tau_E_neq(); // Double contract the two SymmetricTensor
      dissipation_term /= (2 * viscosity_mode_1);

      return dissipation_term.val();
    }

  protected:
    std::vector<double>                 mu_infty;
    std::vector<double>                 alpha_infty;
    std::vector<double>                 mu_mode_1;
    std::vector<double>                 alpha_mode_1;
    double                              viscosity_mode_1;
    SymmetricTensor<2, dim, double>     Cinv_v_1;
    SymmetricTensor<2, dim, double>     Cinv_v_1_converged;
    SymmetricTensor<2, dim, NumberType> tau_neq_1;

    SymmetricTensor<2, dim, NumberType>
    get_tau_E_base(const Tensor<2, dim, NumberType> &F) const override
    {
      return (get_tau_E_neq() + get_tau_E_eq(F));
    }

    SymmetricTensor<2, dim, NumberType>
    get_tau_E_eq(const Tensor<2, dim, NumberType> &F) const
    {
      const SymmetricTensor<2, dim, NumberType> B =
        symmetrize(F * transpose(F));

      // get squared principal stretches (eigenvalues of b) and eigenvectors of
      // b
      std::array<std::pair<NumberType, Tensor<1, dim, NumberType>>, dim>
        eigen_B;
      eigen_B = eigenvectors(B, this->eigen_solver);

      // compute isochoric principal stretches
      double                     det_F = Tensor<0, dim, double>(determinant(F));
      Tensor<1, dim, NumberType> lambda_iso;

      for (unsigned int a = 0; a < dim; ++a)
        {
          lambda_iso[a] =
            std::pow(det_F, -1.0 / 3) * std::sqrt(eigen_B[a].first);
        }

      SymmetricTensor<2, dim, NumberType> tau;

      for (unsigned int A = 0; A < dim; ++A)
        {
          SymmetricTensor<2, dim, NumberType> ev_basis =
            symmetrize(outer_product(eigen_B[A].second, eigen_B[A].second));
          tau += get_beta_infty(lambda_iso, A) * ev_basis;
        }
      return tau;
    }

    SymmetricTensor<2, dim, NumberType>
    get_tau_E_neq() const
    {
      return tau_neq_1;
    }

    NumberType
    get_beta_infty(Tensor<1, dim, NumberType> &lambda_iso,
                   const unsigned int         &A) const
    {
      NumberType beta = 0.0;

      for (unsigned int i = 0; i < 3; ++i) // 3rd-order Ogden model
        {
          NumberType aux = 0.0;
          for (int p = 0; p < dim; ++p)
            aux += std::pow(lambda_iso[p], alpha_infty[i]);

          aux *= -1.0 / dim;
          aux += std::pow(lambda_iso[A], alpha_infty[i]);
          aux *= mu_infty[i];

          beta += aux;
        }
      return beta;
    }

    NumberType
    get_beta_mode_1(std::vector<NumberType> &lambda, const int &A) const
    {
      NumberType beta = 0.0;

      for (unsigned int i = 0; i < 3; ++i) // 3rd-order Ogden model
        {
          NumberType aux = 0.0;
          for (int p = 0; p < dim; ++p)
            aux += std::pow(lambda[p], alpha_mode_1[i]);

          aux *= -1.0 / dim;
          aux += std::pow(lambda[A], alpha_mode_1[i]);
          aux *= mu_mode_1[i];

          beta += aux;
        }
      return beta;
    }

    NumberType
    get_gamma_mode_1(std::vector<NumberType> &lambda,
                     const int               &A,
                     const int               &B) const
    {
      NumberType gamma = 0.0;

      if (A == B)
        {
          for (unsigned int i = 0; i < 3; ++i)
            {
              NumberType aux = 0.0;
              for (int p = 0; p < dim; ++p)
                aux += std::pow(lambda[p], alpha_mode_1[i]);

              aux *= 1.0 / (dim * dim);
              aux += 1.0 / dim * std::pow(lambda[A], alpha_mode_1[i]);
              aux *= mu_mode_1[i] * alpha_mode_1[i];

              gamma += aux;
            }
        }
      else
        {
          for (unsigned int i = 0; i < 3; ++i)
            {
              NumberType aux = 0.0;
              for (int p = 0; p < dim; ++p)
                aux += std::pow(lambda[p], alpha_mode_1[i]);

              aux *= 1.0 / (dim * dim);
              aux -= 1.0 / dim * std::pow(lambda[A], alpha_mode_1[i]);
              aux -= 1.0 / dim * std::pow(lambda[B], alpha_mode_1[i]);
              aux *= mu_mode_1[i] * alpha_mode_1[i];

              gamma += aux;
            }
        }

      return gamma;
    }
  };


  //@sect4{Derived class: Double-mode Ogden viscoelastic material}
  // We use the finite viscoelastic model described in
  // Reese & Govindjee (1998) doi:10.1016/S0020-7683(97)00217-5
  // The algorithm for the implicit exponential time integration is given in
  // Budday et al. (2017) doi: 10.1016/j.actbio.2017.06.024
  template <int dim, typename NumberType = Sacado::Fad::DFad<double>>
  class visco2_Ogden : public Material_Hyperelastic<dim, NumberType>
  {
  public:
    visco2_Ogden(const Parameters::AllParameters &parameters,
                 const std::shared_ptr<Time>      time)
      : Material_Hyperelastic<dim, NumberType>(parameters, time)
      , mu_infty(
          {parameters.mu1_infty, parameters.mu2_infty, parameters.mu3_infty})
      , alpha_infty({parameters.alpha1_infty,
                     parameters.alpha2_infty,
                     parameters.alpha3_infty})
      ,
      // first viscous mode
      mu_mode_1(
        {parameters.mu1_mode_1, parameters.mu2_mode_1, parameters.mu3_mode_1})
      , alpha_mode_1({parameters.alpha1_mode_1,
                      parameters.alpha2_mode_1,
                      parameters.alpha3_mode_1})
      , viscosity_mode_1(parameters.viscosity_mode_1)
      , Cinv_v_1(Physics::Elasticity::StandardTensors<dim>::I)
      , Cinv_v_1_converged(Physics::Elasticity::StandardTensors<dim>::I)
      ,
      // second viscous mode
      mu_mode_2(
        {parameters.mu1_mode_2, parameters.mu2_mode_2, parameters.mu3_mode_2})
      , alpha_mode_2({parameters.alpha1_mode_2,
                      parameters.alpha2_mode_2,
                      parameters.alpha3_mode_2})
      , viscosity_mode_2(parameters.viscosity_mode_2)
      , Cinv_v_2(Physics::Elasticity::StandardTensors<dim>::I)
      , Cinv_v_2_converged(Physics::Elasticity::StandardTensors<dim>::I)
    {}
    virtual ~visco2_Ogden()
    {}

    void
    update_internal_equilibrium(const Tensor<2, dim, NumberType> &F) override
    {
      Material_Hyperelastic<dim, NumberType>::update_internal_equilibrium(F);

      // update right Cauchy-Green strain and compute the elastic left
      // Cauchy-Green trial strain
      this->Cinv_v_1 = this->Cinv_v_1_converged;
      SymmetricTensor<2, dim, NumberType> B_e_1_tr =
        symmetrize(F * this->Cinv_v_1 * transpose(F));
      this->Cinv_v_2 = this->Cinv_v_2_converged;
      SymmetricTensor<2, dim, NumberType> B_e_2_tr =
        symmetrize(F * this->Cinv_v_2 * transpose(F));

      // Compute eigenvalues and eigenvectors of the trail strain tensors and
      // store them in an array of pairs
      const std::array<std::pair<NumberType, Tensor<1, dim, NumberType>>, dim>
        eigen_B_e_1_tr = eigenvectors(B_e_1_tr, this->eigen_solver);
      const std::array<std::pair<NumberType, Tensor<1, dim, NumberType>>, dim>
        eigen_B_e_2_tr = eigenvectors(B_e_2_tr, this->eigen_solver);

      // Compute elastic trial eigenvalues lambda and logarithmic principal
      // stretches epsilon and store them as vectors
      Tensor<1, dim, NumberType> lambdas_e_1_tr, lambdas_e_2_tr;
      Tensor<1, dim, NumberType> epsilon_e_1_tr, epsilon_e_2_tr;

      for (int a = 0; a < dim; ++a)
        {
          lambdas_e_1_tr[a] = std::sqrt(eigen_B_e_1_tr[a].first);
          epsilon_e_1_tr[a] = std::log(lambdas_e_1_tr[a]);
          lambdas_e_2_tr[a] = std::sqrt(eigen_B_e_2_tr[a].first);
          epsilon_e_2_tr[a] = std::log(lambdas_e_2_tr[a]);
        }

      // Define tolerace and check values for the local Newton method.
      // Declare a vector for the residual and a 2nd order tensor for the
      // tangent. Compute the elastic Jacobian based on the elastic left
      // Cauchy-Green trial strain.
      const double               tolerance      = 1e-8;
      double                     residual_check = tolerance * 10.0;
      Tensor<1, dim, NumberType> residual_1, residual_2;
      Tensor<2, dim, NumberType> tangent_1, tangent_2;
      static const SymmetricTensor<2, dim, double> I(
        Physics::Elasticity::StandardTensors<dim>::I);
      NumberType J_e_1 = std::sqrt(determinant(B_e_1_tr));
      NumberType J_e_2 = std::sqrt(determinant(B_e_2_tr));

      // Declare a vector to store the isochoric elastic principal stretches, a
      // 2nd order tensor for the actual left Cauchy-Green strain, an iteration
      // counter, vectors for the actual principal stretches and logarithmic
      // principal stretches (set to its trial values).
      std::vector<NumberType> lambdas_e_1_iso(dim), lambdas_e_2_iso(dim);
      SymmetricTensor<2, dim, NumberType> B_e_1, B_e_2;
      int                                 iteration = 0;

      Tensor<1, dim, NumberType> lambdas_e_1, lambdas_e_2;
      Tensor<1, dim, NumberType> epsilon_e_1, epsilon_e_2;
      epsilon_e_1 = epsilon_e_1_tr;
      epsilon_e_2 = epsilon_e_2_tr;

      while (residual_check > tolerance)
        {
          // Compute the updated elastic Jacobians with the help of auxiliary
          // variables
          NumberType aux_J_e_1 = 1.0;

          for (unsigned int a = 0; a < dim; ++a)
            {
              lambdas_e_1[a] = std::exp(epsilon_e_1[a]);
              aux_J_e_1 *= lambdas_e_1[a];
            }
          J_e_1 = aux_J_e_1;

          // Compute the isochoric elastic principal stretches
          for (unsigned int a = 0; a < dim; ++a)
            lambdas_e_1_iso[a] = lambdas_e_1[a] * std::pow(J_e_1, -1.0 / dim);

          for (unsigned int a = 0; a < dim; ++a)
            {
              residual_1[a] = get_beta_mode_1(lambdas_e_1_iso, a);
              residual_1[a] *=
                this->time->get_delta_t() / (2.0 * viscosity_mode_1);
              residual_1[a] += epsilon_e_1[a];
              residual_1[a] -= epsilon_e_1_tr[a];

              for (unsigned int b = 0; b < dim; ++b)
                {
                  tangent_1[a][b] = get_gamma_mode_1(lambdas_e_1_iso, a, b);
                  tangent_1[a][b] *=
                    this->time->get_delta_t() / (2.0 * viscosity_mode_1);
                  tangent_1[a][b] += I[a][b];
                }
            }
          epsilon_e_1 -= invert(tangent_1) * residual_1;

          residual_check = 0.0;
          for (unsigned int a = 0; a < dim; ++a)
            {
              if (std::abs(residual_1[a]) > residual_check)
                residual_check =
                  std::abs(Tensor<0, dim, double>(residual_1[a]));
            }
          iteration += 1;
          if (iteration > 15)
            AssertThrow(
              false,
              ExcMessage(
                "No convergence in local Newton iteration for the "
                "first viscoelastic exponential time integration algorithm."));
        }

      residual_check = tolerance * 10.0;
      while (residual_check > tolerance)
        {
          // Compute the updated elastic Jacobians with the help of auxiliary
          // variables
          NumberType aux_J_e_2 = 1.0;

          for (unsigned int a = 0; a < dim; ++a)
            {
              lambdas_e_2[a] = std::exp(epsilon_e_2[a]);
              aux_J_e_2 *= lambdas_e_2[a];
            }
          J_e_2 = aux_J_e_2;

          // Compute the isochoric elastic principal stretches
          for (unsigned int a = 0; a < dim; ++a)
            lambdas_e_2_iso[a] = lambdas_e_2[a] * std::pow(J_e_2, -1.0 / dim);

          for (unsigned int a = 0; a < dim; ++a)
            {
              residual_2[a] = get_beta_mode_2(lambdas_e_2_iso, a);
              residual_2[a] *=
                this->time->get_delta_t() / (2.0 * viscosity_mode_2);
              residual_2[a] += epsilon_e_2[a];
              residual_2[a] -= epsilon_e_2_tr[a];

              for (unsigned int b = 0; b < dim; ++b)
                {
                  tangent_2[a][b] = get_gamma_mode_2(lambdas_e_2_iso, a, b);
                  tangent_2[a][b] *=
                    this->time->get_delta_t() / (2.0 * viscosity_mode_2);
                  tangent_2[a][b] += I[a][b];
                }
            }
          epsilon_e_2 -= invert(tangent_2) * residual_2;

          residual_check = 0.0;
          for (unsigned int a = 0; a < dim; ++a)
            {
              if (std::abs(residual_2[a]) > residual_check)
                residual_check =
                  std::abs(Tensor<0, dim, double>(residual_2[a]));
            }
          iteration += 1;
          if (iteration > 15)
            AssertThrow(
              false,
              ExcMessage(
                "No convergence in local Newton iteration for the "
                "second viscoelastic exponential time integration algorithm."));
        }

      NumberType aux_J_e_1 = 1.0, aux_J_e_2 = 1.0;
      for (unsigned int a = 0; a < dim; ++a)
        {
          lambdas_e_1[a] = std::exp(epsilon_e_1[a]);
          aux_J_e_1 *= lambdas_e_1[a];
          lambdas_e_2[a] = std::exp(epsilon_e_2[a]);
          aux_J_e_2 *= lambdas_e_2[a];
        }
      J_e_1 = aux_J_e_1;
      J_e_2 = aux_J_e_2;

      for (unsigned int a = 0; a < dim; ++a)
        {
          lambdas_e_1_iso[a] = lambdas_e_1[a] * std::pow(J_e_1, -1.0 / dim);
          lambdas_e_2_iso[a] = lambdas_e_2[a] * std::pow(J_e_2, -1.0 / dim);
        }

      for (unsigned int a = 0; a < dim; ++a)
        {
          SymmetricTensor<2, dim, NumberType> B_e_1_aux = symmetrize(
            outer_product(eigen_B_e_1_tr[a].second, eigen_B_e_1_tr[a].second));
          B_e_1_aux *= lambdas_e_1[a] * lambdas_e_1[a];
          B_e_1 += B_e_1_aux;
          SymmetricTensor<2, dim, NumberType> B_e_2_aux = symmetrize(
            outer_product(eigen_B_e_2_tr[a].second, eigen_B_e_2_tr[a].second));
          B_e_2_aux *= lambdas_e_2[a] * lambdas_e_2[a];
          B_e_2 += B_e_2_aux;
        }

      Tensor<2, dim, NumberType> Cinv_v_1_AD =
        symmetrize(invert(F) * B_e_1 * invert(transpose(F)));
      Tensor<2, dim, NumberType> Cinv_v_2_AD =
        symmetrize(invert(F) * B_e_2 * invert(transpose(F)));

      this->tau_neq_1 = 0;
      this->tau_neq_2 = 0;
      for (unsigned int a = 0; a < dim; ++a)
        {
          SymmetricTensor<2, dim, NumberType> tau_neq_1_aux = symmetrize(
            outer_product(eigen_B_e_1_tr[a].second, eigen_B_e_1_tr[a].second));
          tau_neq_1_aux *= get_beta_mode_1(lambdas_e_1_iso, a);
          this->tau_neq_1 += tau_neq_1_aux;
          SymmetricTensor<2, dim, NumberType> tau_neq_2_aux = symmetrize(
            outer_product(eigen_B_e_2_tr[a].second, eigen_B_e_2_tr[a].second));
          tau_neq_2_aux *= get_beta_mode_2(lambdas_e_2_iso, a);
          this->tau_neq_2 += tau_neq_2_aux;
        }

      // Print eigenvalues to file
      /*std::ofstream tau_neq;
      tau_neq.open("tau_neq", std::ofstream::app);
      tau_neq << std::setprecision(6) << std::scientific;
      tau_neq << std::setw(16) << this->time->get_current() << ","
          << std::setw(16) << tau_neq_1 << ","
        << std::setw(16) << tau_neq_2 << std::endl;
      tau_neq.close();*/

      // Store history
      for (unsigned int a = 0; a < dim; ++a)
        for (unsigned int b = 0; b < dim; ++b)
          {
            this->Cinv_v_1[a][b] = Tensor<0, dim, double>(Cinv_v_1_AD[a][b]);
            this->Cinv_v_2[a][b] = Tensor<0, dim, double>(Cinv_v_2_AD[a][b]);
          }
    }

    void
    update_end_timestep() override
    {
      Material_Hyperelastic<dim, NumberType>::update_end_timestep();
      this->Cinv_v_1_converged = this->Cinv_v_1;
      this->Cinv_v_2_converged = this->Cinv_v_2;
    }

    double
    get_viscous_dissipation() const override
    {
      NumberType dissipation_term_1 =
        this->tau_neq_1 *
        this->tau_neq_1; // Double contract the two SymmetricTensor
      dissipation_term_1 /= (2 * viscosity_mode_1);
      NumberType dissipation_term_2 =
        this->tau_neq_2 *
        this->tau_neq_2; // Double contract the two SymmetricTensor
      dissipation_term_2 /= (2 * viscosity_mode_2);

      return dissipation_term_1.val() + dissipation_term_2.val();
    }

  protected:
    std::vector<double>                 mu_infty;
    std::vector<double>                 alpha_infty;
    std::vector<double>                 mu_mode_1;
    std::vector<double>                 alpha_mode_1;
    double                              viscosity_mode_1;
    SymmetricTensor<2, dim, double>     Cinv_v_1;
    SymmetricTensor<2, dim, double>     Cinv_v_1_converged;
    SymmetricTensor<2, dim, NumberType> tau_neq_1;
    std::vector<double>                 mu_mode_2;
    std::vector<double>                 alpha_mode_2;
    double                              viscosity_mode_2;
    SymmetricTensor<2, dim, double>     Cinv_v_2;
    SymmetricTensor<2, dim, double>     Cinv_v_2_converged;
    SymmetricTensor<2, dim, NumberType> tau_neq_2;

    SymmetricTensor<2, dim, NumberType>
    get_tau_E_base(const Tensor<2, dim, NumberType> &F) const override
    {
      return (get_tau_E_neq() + get_tau_E_eq(F));
    }

    SymmetricTensor<2, dim, NumberType>
    get_tau_E_eq(const Tensor<2, dim, NumberType> &F) const
    {
      const SymmetricTensor<2, dim, NumberType> B =
        symmetrize(F * transpose(F));

      std::array<std::pair<NumberType, Tensor<1, dim, NumberType>>, dim>
        eigen_B;
      eigen_B = eigenvectors(B, this->eigen_solver);

      SymmetricTensor<2, dim, NumberType>          tau;
      static const SymmetricTensor<2, dim, double> I(
        Physics::Elasticity::StandardTensors<dim>::I);

      for (unsigned int i = 0; i < 3; ++i)
        {
          for (unsigned int A = 0; A < dim; ++A)
            {
              SymmetricTensor<2, dim, NumberType> tau_aux1 =
                symmetrize(outer_product(eigen_B[A].second, eigen_B[A].second));
              tau_aux1 *=
                mu_infty[i] * std::pow(eigen_B[A].first, (alpha_infty[i] / 2.));
              tau += tau_aux1;
            }
          SymmetricTensor<2, dim, NumberType> tau_aux2(I);
          tau_aux2 *= mu_infty[i];
          tau -= tau_aux2;
        }
      return tau;
    }

    SymmetricTensor<2, dim, NumberType>
    get_tau_E_neq() const
    {
      return tau_neq_1 + tau_neq_2;
    }

    NumberType
    get_beta_mode_1(std::vector<NumberType> &lambda, const int &A) const
    {
      NumberType beta = 0.0;
      // 3rd-order Ogden model
      for (unsigned int i = 0; i < 3; ++i)
        {
          NumberType aux = 0.0;
          for (int p = 0; p < dim; ++p)
            aux += std::pow(lambda[p], alpha_mode_1[i]);

          aux *= -1.0 / dim;
          aux += std::pow(lambda[A], alpha_mode_1[i]);
          aux *= mu_mode_1[i];

          beta += aux;
        }
      return beta;
    }

    NumberType
    get_beta_mode_2(std::vector<NumberType> &lambda, const int &A) const
    {
      NumberType beta = 0.0;
      // 3rd-order Ogden model
      for (unsigned int i = 0; i < 3; ++i)
        {
          NumberType aux = 0.0;
          for (int p = 0; p < dim; ++p)
            aux += std::pow(lambda[p], alpha_mode_2[i]);

          aux *= -1.0 / dim;
          aux += std::pow(lambda[A], alpha_mode_2[i]);
          aux *= mu_mode_2[i];

          beta += aux;
        }
      return beta;
    }

    NumberType
    get_gamma_mode_1(std::vector<NumberType> &lambda,
                     const int               &A,
                     const int               &B) const
    {
      NumberType gamma = 0.0;

      if (A == B)
        {
          for (unsigned int i = 0; i < 3; ++i)
            {
              NumberType aux = 0.0;
              for (int p = 0; p < dim; ++p)
                aux += std::pow(lambda[p], alpha_mode_1[i]);

              aux *= 1.0 / (dim * dim);
              aux += 1.0 / dim * std::pow(lambda[A], alpha_mode_1[i]);
              aux *= mu_mode_1[i] * alpha_mode_1[i];

              gamma += aux;
            }
        }
      else
        {
          for (unsigned int i = 0; i < 3; ++i)
            {
              NumberType aux = 0.0;
              for (int p = 0; p < dim; ++p)
                aux += std::pow(lambda[p], alpha_mode_1[i]);

              aux *= 1.0 / (dim * dim);
              aux -= 1.0 / dim * std::pow(lambda[A], alpha_mode_1[i]);
              aux -= 1.0 / dim * std::pow(lambda[B], alpha_mode_1[i]);
              aux *= mu_mode_1[i] * alpha_mode_1[i];

              gamma += aux;
            }
        }
      return gamma;
    }

    NumberType
    get_gamma_mode_2(std::vector<NumberType> &lambda,
                     const int               &A,
                     const int               &B) const
    {
      NumberType gamma = 0.0;

      if (A == B)
        {
          for (unsigned int i = 0; i < 3; ++i)
            {
              NumberType aux = 0.0;
              for (int p = 0; p < dim; ++p)
                aux += std::pow(lambda[p], alpha_mode_2[i]);

              aux *= 1.0 / (dim * dim);
              aux += 1.0 / dim * std::pow(lambda[A], alpha_mode_2[i]);
              aux *= mu_mode_2[i] * alpha_mode_2[i];

              gamma += aux;
            }
        }
      else
        {
          for (unsigned int i = 0; i < 3; ++i)
            {
              NumberType aux = 0.0;
              for (int p = 0; p < dim; ++p)
                aux += std::pow(lambda[p], alpha_mode_2[i]);

              aux *= 1.0 / (dim * dim);
              aux -= 1.0 / dim * std::pow(lambda[A], alpha_mode_2[i]);
              aux -= 1.0 / dim * std::pow(lambda[B], alpha_mode_2[i]);
              aux *= mu_mode_2[i] * alpha_mode_2[i];

              gamma += aux;
            }
        }
      return gamma;
    }
  };


  // @sect3{Constitutive equation for the fluid component of the biphasic
  // material} We consider two slightly different definitions to define the
  // seepage velocity with a Darcy-like law. Ehlers & Eipper 1999,
  // doi:10.1023/A:1006565509095 Markert 2007, doi:10.1007/s11242-007-9107-6 The
  // selection of one or another is made by the user via the parameters file.
  template <int dim, typename NumberType = Sacado::Fad::DFad<double>>
  class Material_Darcy_Fluid
  {
  public:
    Material_Darcy_Fluid(const Parameters::AllParameters &parameters)
      : fluid_type(parameters.fluid_type)
      , n_OS(parameters.solid_vol_frac)
      , initial_intrinsic_permeability(parameters.init_intrinsic_perm)
      , viscosity_FR(parameters.viscosity_FR)
      , initial_darcy_coefficient(parameters.init_darcy_coef)
      , weight_FR(parameters.weight_FR)
      , kappa_darcy(parameters.kappa_darcy)
      , gravity_term(parameters.gravity_term)
      , density_FR(parameters.density_FR)
      , gravity_direction(parameters.gravity_direction)
      , gravity_value(parameters.gravity_value)
    {
      Assert(kappa_darcy >= 0, ExcInternalError());
    }
    ~Material_Darcy_Fluid()
    {}

    Tensor<1, dim, NumberType>
    get_seepage_velocity_current(
      const Tensor<2, dim, NumberType> &F,
      const Tensor<1, dim, NumberType> &grad_p_fluid) const
    {
      const NumberType det_F = determinant(F);
      Assert(det_F > 0.0, ExcInternalError());

      Tensor<2, dim, NumberType> permeability_term;

      if (fluid_type == "Markert")
        permeability_term =
          get_instrinsic_permeability_current(F) / viscosity_FR;

      else if (fluid_type == "Ehlers")
        permeability_term = get_darcy_flow_current(F) / weight_FR;

      else
        AssertThrow(false,
                    ExcMessage(
                      "Material_Darcy_Fluid --> Only Markert "
                      "and Ehlers formulations have been implemented."));

      return (-1.0 * permeability_term *
              det_F // what about this det_F in there?!
              * (grad_p_fluid - get_body_force_FR_current()));

      // try w_F from the theory
      // const NumberType n_F_inv = det_F/(det_F-n_OS);
      // return ( -1.0 * permeability_term * n_F_inv * (grad_p_fluid -
      // get_body_force_FR_current()) );
    }

    double
    get_porous_dissipation(const Tensor<2, dim, NumberType> &F,
                           const Tensor<1, dim, NumberType> &grad_p_fluid) const
    {
      NumberType                 dissipation_term;
      Tensor<1, dim, NumberType> seepage_velocity;
      Tensor<2, dim, NumberType> permeability_term;

      const NumberType det_F = determinant(F);
      Assert(det_F > 0.0, ExcInternalError());

      if (fluid_type == "Markert")
        {
          permeability_term =
            get_instrinsic_permeability_current(F) / viscosity_FR;
          seepage_velocity = get_seepage_velocity_current(F, grad_p_fluid);
        }
      else if (fluid_type == "Ehlers")
        {
          permeability_term = get_darcy_flow_current(F) / weight_FR;
          seepage_velocity  = get_seepage_velocity_current(F, grad_p_fluid);
        }
      else
        AssertThrow(false,
                    ExcMessage(
                      "Material_Darcy_Fluid --> Only Markert and Ehlers "
                      "formulations have been implemented."));

      dissipation_term =
        (invert(permeability_term) * seepage_velocity) * seepage_velocity;
      dissipation_term *= 1.0 / (det_F * det_F);
      return Tensor<0, dim, double>(dissipation_term);
    }

  protected:
    const std::string fluid_type;
    const double      n_OS;
    const double      initial_intrinsic_permeability;
    const double      viscosity_FR;
    const double      initial_darcy_coefficient;
    const double      weight_FR;
    const double      kappa_darcy;
    const bool        gravity_term;
    const double      density_FR;
    const int         gravity_direction;
    const double      gravity_value;

    Tensor<2, dim, NumberType>
    get_instrinsic_permeability_current(
      const Tensor<2, dim, NumberType> &F) const
    {
      static const SymmetricTensor<2, dim, double> I(
        Physics::Elasticity::StandardTensors<dim>::I);
      // SymmetricTensor<2, dim, double> I;
      // I[0][0] = 2; I[1][1] = 1; I[2][2] = 1;
      const Tensor<2, dim, NumberType> initial_instrinsic_permeability_tensor =
        Tensor<2, dim, double>(initial_intrinsic_permeability * I);

      const NumberType det_F = determinant(F);
      Assert(det_F > 0.0, ExcInternalError());

      const NumberType fraction = (det_F - n_OS) / (1 - n_OS);
      return (NumberType(std::pow(fraction, kappa_darcy)) *
              initial_instrinsic_permeability_tensor);
    }

    Tensor<2, dim, NumberType>
    get_darcy_flow_current(const Tensor<2, dim, NumberType> &F) const
    {
      static const SymmetricTensor<2, dim, double> I(
        Physics::Elasticity::StandardTensors<dim>::I);
      const Tensor<2, dim, NumberType> initial_darcy_flow_tensor =
        Tensor<2, dim, double>(initial_darcy_coefficient * I);

      const NumberType det_F = determinant(F);
      Assert(det_F > 0.0, ExcInternalError());

      const NumberType fraction = (1.0 - (n_OS / det_F)) / (1.0 - n_OS);
      return (NumberType(std::pow(fraction, kappa_darcy)) *
              initial_darcy_flow_tensor);
    }

    Tensor<1, dim, NumberType>
    get_body_force_FR_current() const
    {
      Tensor<1, dim, NumberType> body_force_FR_current;

      if (gravity_term == true)
        {
          Tensor<1, dim, NumberType> gravity_vector;
          gravity_vector[gravity_direction] = gravity_value;
          body_force_FR_current             = density_FR * gravity_vector;
        }
      return body_force_FR_current;
    }
  };

  // @sect3{Quadrature point history}
  // As seen in step-18, the <code> PointHistory </code> class offers a method
  // for storing data at the quadrature points.  Here each quadrature point
  // holds a pointer to a material description.  Thus, different material models
  // can be used in different regions of the domain.  Among other data, we
  // choose to store the ``extra" Kirchhoff stress $\boldsymbol{\tau}_E$ and
  // the dissipation values $\mathcal{D}_p$ and $\mathcal{D}_v$.
  template <int dim, typename NumberType = Sacado::Fad::DFad<double>> // double>
  class PointHistory
  {
  public:
    PointHistory()
    {}

    virtual ~PointHistory()
    {}

    void
    setup_lqp(const Parameters::AllParameters &parameters,
              const std::shared_ptr<Time>     &time)
    {
      if (parameters.mat_type == "Neo-Hooke")
        solid_material.reset(new NeoHooke<dim, NumberType>(parameters, time));
      else if (parameters.mat_type == "Neo-Hooke-PS")
        solid_material.reset(
          new NeoHookeEhlers<dim, NumberType>(parameters, time));
      else if (parameters.mat_type == "Neo-Hooke-Ehlers")
        solid_material.reset(
          new NeoHookeEhlers<dim, NumberType>(parameters, time));
      else if (parameters.mat_type == "Ogden")
        solid_material.reset(new Ogden<dim, NumberType>(parameters, time));
      else if (parameters.mat_type == "visco-Ogden")
        solid_material.reset(
          new visco_Ogden<dim, NumberType>(parameters, time));
      else if (parameters.mat_type == "visco2-Ogden")
        solid_material.reset(
          new visco2_Ogden<dim, NumberType>(parameters, time));
      else
        Assert(false, ExcMessage("Material type not implemented"));

      fluid_material.reset(
        new Material_Darcy_Fluid<dim, NumberType>(parameters));
    }

    SymmetricTensor<2, dim, NumberType>
    get_tau_E(const Tensor<2, dim, NumberType> &F) const
    {
      return solid_material->get_tau_E(F);
    }

    SymmetricTensor<2, dim, NumberType>
    get_Cauchy_E(const Tensor<2, dim, NumberType> &F) const
    {
      return solid_material->get_Cauchy_E(F);
    }

    SymmetricTensor<2, dim, NumberType>
    get_Cauchy_E_base(const Tensor<2, dim, NumberType> &F) const
    {
      return solid_material->get_Cauchy_E_base(F);
    }

    SymmetricTensor<2, dim, NumberType>
    get_Cauchy_E_ext_func(const Tensor<2, dim, NumberType> &F) const
    {
      return solid_material->get_Cauchy_E_ext_func(F);
    }

    double
    get_converged_det_F() const
    {
      return solid_material->get_converged_det_F();
    }

    void
    update_end_timestep()
    {
      solid_material->update_end_timestep();
    }

    void
    update_internal_equilibrium(const Tensor<2, dim, NumberType> &F)
    {
      solid_material->update_internal_equilibrium(F);
    }

    double
    get_viscous_dissipation() const
    {
      return solid_material->get_viscous_dissipation();
    }

    Tensor<1, dim, NumberType>
    get_seepage_velocity_current(
      const Tensor<2, dim, NumberType> &F,
      const Tensor<1, dim, NumberType> &grad_p_fluid) const
    {
      return fluid_material->get_seepage_velocity_current(F, grad_p_fluid);
    }

    double
    get_porous_dissipation(const Tensor<2, dim, NumberType> &F,
                           const Tensor<1, dim, NumberType> &grad_p_fluid) const
    {
      return fluid_material->get_porous_dissipation(F, grad_p_fluid);
    }

    Tensor<1, dim, NumberType>
    get_overall_body_force(const Tensor<2, dim, NumberType> &F,
                           const Parameters::AllParameters  &parameters) const
    {
      Tensor<1, dim, NumberType> body_force;

      if (parameters.gravity_term == true)
        {
          const NumberType det_F_AD = determinant(F);
          Assert(det_F_AD > 0.0, ExcInternalError());

          const NumberType overall_density_ref =
            parameters.density_SR * parameters.solid_vol_frac +
            parameters.density_FR * (det_F_AD - parameters.solid_vol_frac);

          Tensor<1, dim, NumberType> gravity_vector;
          gravity_vector[parameters.gravity_direction] =
            parameters.gravity_value;
          body_force = overall_density_ref * gravity_vector;
        }

      return body_force;
    }

  private:
    std::shared_ptr<Material_Hyperelastic<dim, NumberType>> solid_material;
    std::shared_ptr<Material_Darcy_Fluid<dim, NumberType>>  fluid_material;
  };

  // @sect3{Nonlinear poro-viscoelastic solid}
  // The Solid class is the central class as it represents the problem at hand:
  // the nonlinear poro-viscoelastic solid
  template <int dim>
  class Solid
  {
  public:
    Solid(const Parameters::AllParameters &parameters);
    virtual ~Solid();
    void
    run();

  protected:
    using ADNumberType = Sacado::Fad::DFad<double>;

    std::ofstream outfile;
    std::ofstream pointfile;
    std::ofstream nodefile;

    struct PerTaskData_ASM;
    template <typename NumberType = double>
    struct ScratchData_ASM;

    // Generate mesh
    virtual void
    make_grid() = 0;

    // Define points for post-processing
    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) = 0;

    // Set up the finite element system to be solved:
    void
    system_setup(TrilinosWrappers::MPI::BlockVector &solution_delta_OUT);

    // Extract sub-blocks from the global matrix
    void
    determine_component_extractors();

    // Several functions to assemble the system and right hand side matrices
    // using multithreading.
    void
    assemble_system(
      const TrilinosWrappers::MPI::BlockVector &solution_delta_OUT);
    void
    assemble_system_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM<ADNumberType>                        &scratch,
      PerTaskData_ASM                                      &data) const;
    void
    copy_local_to_global_system(const PerTaskData_ASM &data);

    // Define boundary conditions
    virtual void
    make_constraints(const int &it_nr);
    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) = 0;
    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const = 0;
    virtual double
    get_prescribed_fluid_flow(const types::boundary_id &boundary_id,
                              const Point<dim>         &pt) const = 0;
    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const = 0;
    virtual std::pair<types::boundary_id, types::boundary_id>
    get_drained_boundary_id_for_output() const = 0;
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const = 0;

    // Create and update the quadrature points.
    void
    setup_qph();

    // Solve non-linear system using a Newton-Raphson scheme
    void
    solve_nonlinear_timestep(
      TrilinosWrappers::MPI::BlockVector &solution_delta_OUT);

    // Solve the linearized equations using a direct solver
    void
    solve_linear_system(TrilinosWrappers::MPI::BlockVector &newton_update_OUT);

    // Retrieve the  solution
    TrilinosWrappers::MPI::BlockVector
    get_total_solution(
      const TrilinosWrappers::MPI::BlockVector &solution_delta_IN) const;

    // Store the converged values of the internal variables at the end of each
    // timestep
    // void update_end_timestep();
    double
    update_end_timestep();
    double
    jacobian_on_faces(TrilinosWrappers::MPI::BlockVector solution);

    // Post-processing and writing data to files
    void
    output_results_to_vtu(const unsigned int                 timestep,
                          const double                       current_time,
                          TrilinosWrappers::MPI::BlockVector solution) const;
    void
    output_bcs_to_vtu(const unsigned int                 timestep,
                      const double                       current_time,
                      TrilinosWrappers::MPI::BlockVector solution) const;
    void
    output_results_to_plot(const unsigned int                 timestep,
                           const double                       current_time,
                           TrilinosWrappers::MPI::BlockVector solution,
                           std::vector<Point<dim>>           &tracked_vertices,
                           std::ofstream                     &pointfile) const;
    void
    output_results_averaged_on_nodes(
      const unsigned int                 timestep,
      const double                       current_time,
      TrilinosWrappers::MPI::BlockVector solution,
      std::vector<Point<dim>>           &tracked_vertices,
      std::ofstream                     &nodefile) const;

    // Headers and footer for the output files
    void
    print_console_file_header(std::ofstream &outfile) const;
    void
    print_plot_file_header(std::vector<Point<dim>> &tracked_vertices,
                           std::ofstream           &pointfile) const;
    void
    print_console_file_footer(std::ofstream &outfile) const;
    void
    print_plot_file_footer(std::ofstream &pointfile) const;

    // For parallel communication
    MPI_Comm                   mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    mutable ConditionalOStream pcout;

    // A collection of the parameters used to describe the problem setup
    const Parameters::AllParameters &parameters;

    // Declare an instance of dealii Triangulation class (mesh)
    parallel::shared::Triangulation<dim> triangulation;

    // Keep track of the current time and the time spent evaluating certain
    // functions
    std::shared_ptr<Time> time;
    TimerOutput           timerconsole;
    TimerOutput           timerfile;

    // A storage object for quadrature point information.
    CellDataStorage<typename Triangulation<dim>::active_cell_iterator,
                    PointHistory<dim, ADNumberType>>
      quadrature_point_history;

    // Integers to store polynomial degree (needed for output)
    const unsigned int degree_displ;
    const unsigned int degree_pore;

    // Declare an instance of dealii FESystem class (finite element definition)
    const FESystem<dim> fe;

    // Mapping for curved boundaries
    const MappingQ<dim> mapping;

    // Declare an instance of dealii DoFHandler class (assign DoFs to mesh)
    DoFHandler<dim> dof_handler_ref;

    // Integer to store DoFs per element (this value will be used often)
    const unsigned int dofs_per_cell;

    // Declare an instance of dealii Extractor objects used to retrieve
    // information from the solution vectors We will use "u_fe" and
    // "p_fluid_fe"as subscript in operator [] expressions on FEValues and
    // FEFaceValues objects to extract the components of the displacement vector
    // and fluid pressure, respectively.
    const FEValuesExtractors::Vector u_fe;
    const FEValuesExtractors::Scalar p_fluid_fe;

    // Description of how the block-system is arranged. There are 3 blocks:
    //   0 - vector DOF displacements u
    //   1 - scalar DOF fluid pressure p_fluid
    static const unsigned int n_blocks          = 2;
    static const unsigned int n_components      = dim + 1;
    static const unsigned int first_u_component = 0;
    static const unsigned int p_fluid_component = dim;

    enum
    {
      u_block       = 0,
      p_fluid_block = 1
    };

    // Extractors
    const FEValuesExtractors::Scalar x_displacement;
    const FEValuesExtractors::Scalar y_displacement;
    const FEValuesExtractors::Scalar z_displacement;
    const FEValuesExtractors::Scalar pressure;

    // Block data
    std::vector<unsigned int> block_component;

    // DoF index data
    std::vector<IndexSet> all_locally_owned_dofs;
    IndexSet              locally_owned_dofs;
    IndexSet              locally_relevant_dofs;
    std::vector<IndexSet> locally_owned_partitioning;
    std::vector<IndexSet> locally_relevant_partitioning;

    std::vector<types::global_dof_index> dofs_per_block;
    std::vector<types::global_dof_index> element_indices_u;
    std::vector<types::global_dof_index> element_indices_p_fluid;

    // Declare an instance of dealii QGauss class (The Gauss-Legendre family of
    // quadrature rules for numerical integration) Gauss Points in element, with
    // n quadrature points (in each space direction <dim> )
    const QGauss<dim> qf_cell;
    // const QGaussLobatto<dim>                qf_cell;
    // const QGaussLobattoChebyshev<dim>                qf_cell;
    // Gauss Points on element faces (used for definition of BCs)
    const QGauss<dim - 1> qf_face;
    // const QGaussLobatto<dim - 1>            qf_face;
    // const QGaussLobattoChebyshev<dim - 1>            qf_face;
    // Integer to store num GPs per element (this value will be used often)
    const unsigned int n_q_points;
    // Integer to store num GPs per face (this value will be used often)
    const unsigned int n_q_points_f;

    // Declare an instance of dealii AffineConstraints class (linear constraints
    // on DoFs due to hanging nodes or BCs)
    AffineConstraints<double> constraints;
    AffineConstraints<double> hanging_node_constraints;
    enum AffineConstraints<double>::MergeConflictBehavior dirichlet_wins;

    // An index set to deal with the contact problem
    IndexSet active_set;

    // Declare an instance of dealii classes necessary for FE system set-up and
    // assembly Store elements of tangent matrix (indicated by SparsityPattern
    // class) as sparse matrix (more efficient)
    TrilinosWrappers::BlockSparseMatrix tangent_matrix;
    TrilinosWrappers::BlockSparseMatrix tangent_matrix_preconditioner;
    // Right hand side vector of forces
    TrilinosWrappers::MPI::BlockVector system_rhs;
    // Total displacement values + pressure (accumulated solution to FE system)
    TrilinosWrappers::MPI::BlockVector solution_n;
    TrilinosWrappers::MPI::BlockVector distributed_solution;

    // Non-block system for the direct solver. We will copy the block system
    // into these to solve the linearized system of equations.
    TrilinosWrappers::SparseMatrix tangent_matrix_nb;
    TrilinosWrappers::MPI::Vector  system_rhs_nb;

    // We define variables to store norms and update norms and normalisation
    // factors.
    struct Errors
    {
      Errors()
        : norm(1.0)
        , u(1.0)
        , p_fluid(1.0)
      {}

      void
      reset()
      {
        norm    = 1.0;
        u       = 1.0;
        p_fluid = 1.0;
      }
      void
      normalise(const Errors &rhs)
      {
        if (rhs.norm != 0.0)
          norm /= rhs.norm;
        if (rhs.u != 0.0)
          u /= rhs.u;
        if (rhs.p_fluid != 0.0)
          p_fluid /= rhs.p_fluid;
      }

      double norm, u, p_fluid;
    };

    // Declare several instances of the "Error" structure
    Errors error_residual, error_residual_0, error_residual_norm, error_update,
      error_update_0, error_update_norm, error_residual_norm_last,
      error_update_norm_last;

    // Methods to calculate error measures
    void
    get_error_residual(Errors &error_residual_OUT);
    void
    get_error_update(const TrilinosWrappers::MPI::BlockVector &newton_update_IN,
                     Errors &error_update_OUT);

    // Print information to screen
    void
    print_conv_header();
    void
    print_conv_footer();

    // NOTE: In all functions, we pass by reference (&), so these functions work
    // on the original copy (not a clone copy),
    //       modifying the input variables inside the functions will change them
    //       outside the function.
  };

  // @sect3{Implementation of the <code>Solid</code> class}
  // @sect4{Public interface}
  // We initialise the Solid class using data extracted from the parameter file.
  template <int dim>
  Solid<dim>::Solid(const Parameters::AllParameters &parameters)
    : mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , pcout(std::cout, this_mpi_process == 0)
    , parameters(parameters)
    , triangulation(mpi_communicator, Triangulation<dim>::maximum_smoothing)
    , timerconsole(mpi_communicator,
                   pcout,
                   TimerOutput::summary,
                   TimerOutput::wall_times)
    , timerfile(mpi_communicator,
                outfile,
                TimerOutput::summary,
                TimerOutput::wall_times)
    , degree_displ(parameters.poly_degree_displ)
    , degree_pore(parameters.poly_degree_pore)
    , fe(FE_Q<dim>(parameters.poly_degree_displ),
         dim,
         FE_Q<dim>(parameters.poly_degree_pore),
         1)
    , mapping(1)
    , dof_handler_ref(triangulation)
    , dofs_per_cell(fe.dofs_per_cell)
    , u_fe(first_u_component)
    , p_fluid_fe(p_fluid_component)
    , x_displacement(first_u_component)
    , y_displacement(first_u_component + 1)
    , z_displacement(first_u_component + 2)
    , pressure(p_fluid_component)
    , dofs_per_block(n_blocks)
    , qf_cell(parameters.quad_order)
    , qf_face(parameters.quad_order)
    , n_q_points(qf_cell.size())
    , n_q_points_f(qf_face.size())
  {
    Assert(dim == 3,
           ExcMessage("This problem only works in 3 space dimensions."));
    determine_component_extractors();

    if (parameters.input_file.empty())
      {
        this->time = std::make_shared<TimeFixed>(parameters.end_time,
                                                 parameters.end_load_time,
                                                 parameters.delta_t);
      }
    else
      {
        this->time = std::make_shared<TimeFile>(parameters.input_directory +
                                                '/' + parameters.input_file);
      }
  }

  // The class destructor simply clears the data held by the DOFHandler
  template <int dim>
  Solid<dim>::~Solid()
  {
    dof_handler_ref.clear();
  }

  // Runs the 3D solid problem
  template <int dim>
  void
  Solid<dim>::run()
  {
    // The current solution increment is defined as a block vector to reflect
    // the structure of the PDE system, with multiple solution components
    TrilinosWrappers::MPI::BlockVector solution_delta;

    // Open file
    if (this_mpi_process == 0)
      {
        outfile.open(parameters.output_directory + "/console-output.sol");
        print_console_file_header(outfile);
      }

    // Generate mesh
    make_grid();

    // Assign DOFs and create the stiffness and right-hand-side force vector
    system_setup(solution_delta);

    // Define points for post-processing
    std::vector<Point<dim>> tracked_vertices(2);
    define_tracked_vertices(tracked_vertices);
    std::vector<Point<dim>> reaction_force;

    if (this_mpi_process == 0)
      {
        pointfile.open(parameters.output_directory + "/data-for-gnuplot.sol");
        print_plot_file_header(tracked_vertices, pointfile);
      }

    // Print results to output file
    if (parameters.outfiles_requested == "all")
      {
        output_results_to_vtu(time->get_timestep(),
                              time->get_current(),
                              solution_n);
        output_bcs_to_vtu(time->get_timestep(),
                          time->get_current(),
                          solution_n);
      }
    else if (parameters.outfiles_requested == "solution")
      {
        output_results_to_vtu(time->get_timestep(),
                              time->get_current(),
                              solution_n);
      }
    else if (parameters.outfiles_requested == "bcs")
      {
        output_bcs_to_vtu(time->get_timestep(),
                          time->get_current(),
                          solution_n);
      }

    output_results_to_plot(time->get_timestep(),
                           time->get_current(),
                           solution_n,
                           tracked_vertices,
                           pointfile);

    // Increment time step (=load step)
    // NOTE: In solving the quasi-static problem, the time becomes a loading
    // parameter, i.e. we increase the loading linearly with time, making the
    // two concepts interchangeable.
    time->increment_time(1);

    // Print information on screen
    pcout << "\nSolver:";
    pcout << "\n  CST     = make constraints";
    pcout << "\n  ASM_SYS = assemble system";
    pcout << "\n  SLV     = linear solver \n";

    // Print information on file
    outfile << "\nSolver:";
    outfile << "\n  CST     = make constraints";
    outfile << "\n  ASM_SYS = assemble system";
    outfile << "\n  SLV     = linear solver \n";


    double det_F_min     = 1.0;
    double det_F_min_mpi = 1.0;
    // double det_F_change     = 1.0;
    // double det_F_change_mpi = 1.0;

    while ((time->get_end() - time->get_current()) > -1.0 * parameters.tol_u)
      {
        // Initialize the current solution increment to zero
        solution_delta = 0.0;

        this->pcout << "J_min_change = " << det_F_min << std::endl;
        // std::cout << "J = " << det_F_min_mpi << std::endl;
        // Solve the non-linear system using a Newton-Raphson scheme
        solve_nonlinear_timestep(solution_delta);

        // Add the computed solution increment to total solution
        solution_n += solution_delta;


        // det_F_change_mpi = jacobian_on_faces(solution_n);
        // det_F_change = Utilities::MPI::max(det_F_change_mpi,
        // mpi_communicator); Store the converged values of the internal
        // variables
        det_F_min_mpi = update_end_timestep();
        det_F_min     = Utilities::MPI::max(det_F_min_mpi, mpi_communicator);
        // det_F_min = std::max(det_F_change, det_F_min);

        // Output results
        if ((time->get_timestep() % parameters.timestep_output) == 0)
          {
            if (parameters.outfiles_requested == "all")
              {
                output_results_to_vtu(time->get_timestep(),
                                      time->get_current(),
                                      solution_n);
                output_bcs_to_vtu(time->get_timestep(),
                                  time->get_current(),
                                  solution_n);
              }
            else if (parameters.outfiles_requested == "solution")
              {
                output_results_to_vtu(time->get_timestep(),
                                      time->get_current(),
                                      solution_n);
              }
            else if (parameters.outfiles_requested == "bcs")
              {
                output_bcs_to_vtu(time->get_timestep(),
                                  time->get_current(),
                                  solution_n);
              }
          }

        output_results_to_plot(time->get_timestep(),
                               time->get_current(),
                               solution_n,
                               tracked_vertices,
                               pointfile);

        // Increment the time step (=load step)
        time->increment_time(det_F_min);
      }

    // Print the footers and close files
    if (this_mpi_process == 0)
      {
        // print_plot_file_footer(pointfile)
        pointfile.close();
        // print_console_file_footer(outfile);

        // NOTE: ideally, we should close the outfile here [ >> outfile.close
        // (); ] But if we do, then the timer output will not be printed. That
        // is why we leave it open.
      }
  }

  // @sect4{Private interface}
  // We define the structures needed for parallelization with Threading Building
  // Blocks (TBB) Tangent matrix and right-hand side force vector assembly
  // structures. PerTaskData_ASM stores local contributions
  template <int dim>
  struct Solid<dim>::PerTaskData_ASM
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    PerTaskData_ASM(const unsigned int dofs_per_cell)
      : cell_matrix(dofs_per_cell, dofs_per_cell)
      , cell_rhs(dofs_per_cell)
      , local_dof_indices(dofs_per_cell)
    {}

    void
    reset()
    {
      cell_matrix = 0.0;
      cell_rhs    = 0.0;
    }
  };

  // ScratchData_ASM stores larger objects used during the assembly
  template <int dim>
  template <typename NumberType>
  struct Solid<dim>::ScratchData_ASM
  {
    const TrilinosWrappers::MPI::BlockVector &solution_total;

    // Integration helper
    FEValues<dim>     fe_values_ref;
    FEFaceValues<dim> fe_face_values_ref;

    // Quadrature point solution
    std::vector<NumberType>                 local_dof_values;
    std::vector<Tensor<2, dim, NumberType>> solution_grads_u_total;
    std::vector<NumberType>                 solution_values_p_fluid_total;
    std::vector<Tensor<1, dim, NumberType>> solution_grads_p_fluid_total;
    std::vector<Tensor<1, dim, NumberType>> solution_grads_face_p_fluid_total;

    // shape function values
    std::vector<std::vector<Tensor<1, dim>>> Nx;
    std::vector<std::vector<double>>         Nx_p_fluid;
    // shape function gradients
    std::vector<std::vector<Tensor<2, dim, NumberType>>>          grad_Nx;
    std::vector<std::vector<SymmetricTensor<2, dim, NumberType>>> symm_grad_Nx;
    std::vector<std::vector<Tensor<1, dim, NumberType>>> grad_Nx_p_fluid;

    ScratchData_ASM(
      const FiniteElement<dim> &fe_cell,
      const QGauss<dim>        &qf_cell,
      const UpdateFlags         uf_cell,
      const QGauss<dim - 1>    &qf_face,
      const UpdateFlags         uf_face,
      // const QGaussLobatto<dim> &qf_cell, const UpdateFlags uf_cell,
      // const QGaussLobatto<dim - 1> & qf_face, const UpdateFlags uf_face,
      // const QGaussLobattoChebyshev<dim> &qf_cell, const UpdateFlags uf_cell,
      // const QGaussLobattoChebyshev<dim - 1> & qf_face, const UpdateFlags
      // uf_face,
      const TrilinosWrappers::MPI::BlockVector &solution_total)
      : solution_total(solution_total)
      , fe_values_ref(fe_cell, qf_cell, uf_cell)
      , fe_face_values_ref(fe_cell, qf_face, uf_face)
      , local_dof_values(fe_cell.dofs_per_cell)
      , solution_grads_u_total(qf_cell.size())
      , solution_values_p_fluid_total(qf_cell.size())
      , solution_grads_p_fluid_total(qf_cell.size())
      , solution_grads_face_p_fluid_total(qf_face.size())
      , Nx(qf_cell.size(), std::vector<Tensor<1, dim>>(fe_cell.dofs_per_cell))
      , Nx_p_fluid(qf_cell.size(), std::vector<double>(fe_cell.dofs_per_cell))
      , grad_Nx(qf_cell.size(),
                std::vector<Tensor<2, dim, NumberType>>(fe_cell.dofs_per_cell))
      , symm_grad_Nx(qf_cell.size(),
                     std::vector<SymmetricTensor<2, dim, NumberType>>(
                       fe_cell.dofs_per_cell))
      , grad_Nx_p_fluid(qf_cell.size(),
                        std::vector<Tensor<1, dim, NumberType>>(
                          fe_cell.dofs_per_cell))
    {}

    ScratchData_ASM(const ScratchData_ASM &rhs)
      : solution_total(rhs.solution_total)
      , fe_values_ref(rhs.fe_values_ref.get_fe(),
                      rhs.fe_values_ref.get_quadrature(),
                      rhs.fe_values_ref.get_update_flags())
      , fe_face_values_ref(rhs.fe_face_values_ref.get_fe(),
                           rhs.fe_face_values_ref.get_quadrature(),
                           rhs.fe_face_values_ref.get_update_flags())
      , local_dof_values(rhs.local_dof_values)
      , solution_grads_u_total(rhs.solution_grads_u_total)
      , solution_values_p_fluid_total(rhs.solution_values_p_fluid_total)
      , solution_grads_p_fluid_total(rhs.solution_grads_p_fluid_total)
      , solution_grads_face_p_fluid_total(rhs.solution_grads_face_p_fluid_total)
      , Nx(rhs.Nx)
      , Nx_p_fluid(rhs.Nx_p_fluid)
      , grad_Nx(rhs.grad_Nx)
      , symm_grad_Nx(rhs.symm_grad_Nx)
      , grad_Nx_p_fluid(rhs.grad_Nx_p_fluid)
    {}

    void
    reset()
    {
      const unsigned int n_q_points      = Nx_p_fluid.size();
      const unsigned int n_dofs_per_cell = Nx_p_fluid[0].size();

      Assert(local_dof_values.size() == n_dofs_per_cell, ExcInternalError());

      for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
        {
          local_dof_values[k] = 0.0;
        }

      Assert(solution_grads_u_total.size() == n_q_points, ExcInternalError());
      Assert(solution_values_p_fluid_total.size() == n_q_points,
             ExcInternalError());
      Assert(solution_grads_p_fluid_total.size() == n_q_points,
             ExcInternalError());

      Assert(Nx.size() == n_q_points, ExcInternalError());
      Assert(grad_Nx.size() == n_q_points, ExcInternalError());
      Assert(symm_grad_Nx.size() == n_q_points, ExcInternalError());

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
          Assert(grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());
          Assert(symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          solution_grads_u_total[q_point]        = 0.0;
          solution_values_p_fluid_total[q_point] = 0.0;
          solution_grads_p_fluid_total[q_point]  = 0.0;

          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              Nx[q_point][k]              = 0.0;
              Nx_p_fluid[q_point][k]      = 0.0;
              grad_Nx[q_point][k]         = 0.0;
              symm_grad_Nx[q_point][k]    = 0.0;
              grad_Nx_p_fluid[q_point][k] = 0.0;
            }
        }

      const unsigned int n_f_q_points =
        solution_grads_face_p_fluid_total.size();
      Assert(solution_grads_face_p_fluid_total.size() == n_f_q_points,
             ExcInternalError());

      for (unsigned int f_q_point = 0; f_q_point < n_f_q_points; ++f_q_point)
        solution_grads_face_p_fluid_total[f_q_point] = 0.0;
    }
  };

  // Define the boundary conditions on the mesh
  template <int dim>
  void
  Solid<dim>::make_constraints(const int &it_nr_IN)
  {
    pcout << " CST " << std::flush;
    outfile << " CST " << std::flush;

    if (it_nr_IN > 1)
      return;


    const bool apply_dirichlet_bc = (it_nr_IN == 0);

    if (apply_dirichlet_bc)
      {
        //////////////////////// TEST /////////////////////////
        const UpdateFlags uf_cell(update_quadrature_points | update_values |
                                  update_JxW_values);
        FEValues<dim>     fe_values_ref(mapping, fe, qf_cell, uf_cell);
        //////////////////////// TEST /////////////////////////

        for (auto cell : this->triangulation.active_cell_iterators())
          {
            //////////////////////// TEST /////////////////////////
            fe_values_ref.reinit(cell);
            //////////////////////// TEST /////////////////////////

            const UpdateFlags uf_face(update_quadrature_points |
                                      update_normal_vectors | update_values |
                                      update_JxW_values);
            FEFaceValues<dim> fe_face_values_ref(mapping, fe, qf_face, uf_face);

            // Start loop over faces in element
            for (unsigned int face = 0;
                 face < GeometryInfo<dim>::faces_per_cell;
                 ++face)
              {
                if (cell->face(face)->at_boundary() == true)
                  {
                    fe_face_values_ref.reinit(cell, face);
                  }
              }
          }
        constraints.clear();
        make_dirichlet_constraints(constraints);
      }
    else
      {
        // AffineConstraints<double> homogeneous_constraints(constraints);
        // //remove
        for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
          if (constraints.is_inhomogeneously_constrained(i) ==
              true)                                // remove: homogeneous_
            constraints.set_inhomogeneity(i, 0.0); // remove: homogeneous_
        // constraints.clear(); //remove
        // constraints.copy_from(homogeneous_constraints); //remove
      }

    constraints.close();
    dirichlet_wins =
      AffineConstraints<double>::MergeConflictBehavior::left_object_wins;
    constraints.merge(hanging_node_constraints, dirichlet_wins);
  }

  // Set-up the FE system
  template <int dim>
  void
  Solid<dim>::system_setup(
    TrilinosWrappers::MPI::BlockVector &solution_delta_OUT)
  {
    timerconsole.enter_subsection("Setup system");
    timerfile.enter_subsection("Setup system");

    // Determine number of components per block
    std::vector<unsigned int> block_component(n_components, u_block);
    block_component[p_fluid_component] = p_fluid_block;

    // The DOF handler is initialised and we renumber the grid in an efficient
    // manner.
    dof_handler_ref.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler_ref);
    DoFRenumbering::component_wise(dof_handler_ref, block_component);

    // hanging_node_constraints.clear();
    // DoFTools::make_hanging_node_constraints(dof_handler_ref,hanging_node_constraints);
    // hanging_node_constraints.close();

    // Count the number of DoFs in each block
    // dofs_per_block.clear();
    // dofs_per_block.resize(n_blocks);
    // DoFTools::count_dofs_per_block(dof_handler_ref, dofs_per_block,
    // block_component);

    dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler_ref, block_component);

    // Setup the sparsity pattern and tangent matrix
    all_locally_owned_dofs =
      DoFTools::locally_owned_dofs_per_subdomain(dof_handler_ref);
    std::vector<IndexSet> all_locally_relevant_dofs =
      DoFTools::locally_relevant_dofs_per_subdomain(dof_handler_ref);

    locally_owned_dofs.clear();
    locally_owned_partitioning.clear();
    Assert(all_locally_owned_dofs.size() > this_mpi_process,
           ExcInternalError());
    locally_owned_dofs = all_locally_owned_dofs[this_mpi_process];

    locally_relevant_dofs.clear();
    locally_relevant_partitioning.clear();
    Assert(all_locally_relevant_dofs.size() > this_mpi_process,
           ExcInternalError());
    locally_relevant_dofs = all_locally_relevant_dofs[this_mpi_process];

    locally_owned_partitioning.reserve(n_blocks);
    locally_relevant_partitioning.reserve(n_blocks);

    for (unsigned int b = 0; b < n_blocks; ++b)
      {
        const types::global_dof_index idx_begin =
          std::accumulate(dofs_per_block.begin(),
                          std::next(dofs_per_block.begin(), b),
                          0);
        const types::global_dof_index idx_end =
          std::accumulate(dofs_per_block.begin(),
                          std::next(dofs_per_block.begin(), b + 1),
                          0);
        locally_owned_partitioning.push_back(
          locally_owned_dofs.get_view(idx_begin, idx_end));
        locally_relevant_partitioning.push_back(
          locally_relevant_dofs.get_view(idx_begin, idx_end));
      }

    // Print information on screen
    pcout << "\nTriangulation:\n"
          << "  Number of active cells: " << triangulation.n_active_cells()
          << " (by partition:";
    for (unsigned int p = 0; p < n_mpi_processes; ++p)
      pcout << (p == 0 ? ' ' : '+')
            << (GridTools::count_cells_with_subdomain_association(triangulation,
                                                                  p));
    pcout << ")" << std::endl;
    pcout << "  Number of degrees of freedom: " << dof_handler_ref.n_dofs()
          << " (by partition:";
    for (unsigned int p = 0; p < n_mpi_processes; ++p)
      pcout << (p == 0 ? ' ' : '+')
            << (DoFTools::count_dofs_with_subdomain_association(dof_handler_ref,
                                                                p));
    pcout << ")" << std::endl;
    pcout << "  Number of degrees of freedom per block: "
          << "[n_u, n_p_fluid] = [" << dofs_per_block[u_block] << ", "
          << dofs_per_block[p_fluid_block] << "]" << std::endl;

    // Print information to file
    outfile << "\nTriangulation:\n"
            << "  Number of active cells: " << triangulation.n_active_cells()
            << " (by partition:";
    for (unsigned int p = 0; p < n_mpi_processes; ++p)
      outfile << (p == 0 ? ' ' : '+')
              << (GridTools::count_cells_with_subdomain_association(
                   triangulation, p));
    outfile << ")" << std::endl;
    outfile << "  Number of degrees of freedom: " << dof_handler_ref.n_dofs()
            << " (by partition:";
    for (unsigned int p = 0; p < n_mpi_processes; ++p)
      outfile << (p == 0 ? ' ' : '+')
              << (DoFTools::count_dofs_with_subdomain_association(
                   dof_handler_ref, p));
    outfile << ")" << std::endl;
    outfile << "  Number of degrees of freedom per block: "
            << "[n_u, n_p_fluid] = [" << dofs_per_block[u_block] << ", "
            << dofs_per_block[p_fluid_block] << "]" << std::endl;

    // We optimise the sparsity pattern to reflect this structure and prevent
    // unnecessary data creation for the right-diagonal block components.
    Table<2, DoFTools::Coupling> coupling(n_components, n_components);
    for (unsigned int ii = 0; ii < n_components; ++ii)
      for (unsigned int jj = 0; jj < n_components; ++jj)

        // Identify "zero" matrix components of FE-system (The two components do
        // not couple)
        if (((ii == p_fluid_component) && (jj < p_fluid_component)) ||
            ((ii < p_fluid_component) && (jj == p_fluid_component)))
          coupling[ii][jj] = DoFTools::none;

        // The rest of components always couple
        else
          coupling[ii][jj] = DoFTools::always;

    TrilinosWrappers::BlockSparsityPattern bsp(locally_owned_partitioning,
                                               mpi_communicator);

    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_ref,
                                            hanging_node_constraints);
    hanging_node_constraints.close();

    DoFTools::make_sparsity_pattern(
      dof_handler_ref, bsp, hanging_node_constraints, false, this_mpi_process);
    bsp.compress();

    // Reinitialize the (sparse) tangent matrix with the given sparsity pattern.
    tangent_matrix.reinit(bsp);

    // Initialize the right hand side and solution vectors with number of DoFs
    system_rhs.reinit(locally_owned_partitioning, mpi_communicator);
    solution_n.reinit(locally_owned_partitioning, mpi_communicator);
    solution_delta_OUT.reinit(locally_owned_partitioning, mpi_communicator);
    distributed_solution.reinit(locally_owned_partitioning, mpi_communicator);

    // Initialize the active set
    active_set.clear();
    active_set.set_size(dof_handler_ref.n_dofs());

    // Non-block system
    TrilinosWrappers::SparsityPattern sp(locally_owned_dofs, mpi_communicator);
    DoFTools::make_sparsity_pattern(
      dof_handler_ref, sp, hanging_node_constraints, false, this_mpi_process);
    sp.compress();
    tangent_matrix_nb.reinit(sp);
    system_rhs_nb.reinit(locally_owned_dofs, mpi_communicator);

    // Set up the quadrature point history
    setup_qph();

    timerconsole.leave_subsection();
    timerfile.leave_subsection();
  }

  // Component extractors: used to extract sub-blocks from the global matrix
  // Description of which local element DOFs are attached to which block
  // component
  template <int dim>
  void
  Solid<dim>::determine_component_extractors()
  {
    element_indices_u.clear();
    element_indices_p_fluid.clear();

    for (unsigned int k = 0; k < fe.dofs_per_cell; ++k)
      {
        const unsigned int k_group = fe.system_to_base_index(k).first.first;
        if (k_group == u_block)
          element_indices_u.push_back(k);
        else if (k_group == p_fluid_block)
          element_indices_p_fluid.push_back(k);
        else
          {
            Assert(k_group <= p_fluid_block, ExcInternalError());
          }
      }
  }

  // Set-up quadrature point history (QPH) data objects
  template <int dim>
  void
  Solid<dim>::setup_qph()
  {
    pcout << "\nSetting up quadrature point data..." << std::endl;
    outfile << "\nSetting up quadrature point data..." << std::endl;

    // Create QPH data objects.
    quadrature_point_history.initialize(triangulation.begin_active(),
                                        triangulation.end(),
                                        n_q_points);

    // Setup the initial quadrature point data using the info stored in
    // parameters
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator> cell(
      IteratorFilters::LocallyOwnedCell(), dof_handler_ref.begin_active()),
      endc(IteratorFilters::LocallyOwnedCell(), dof_handler_ref.end());
    for (; cell != endc; ++cell)
      {
        Assert(cell->is_locally_owned(), ExcInternalError());
        Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

        const std::vector<std::shared_ptr<PointHistory<dim, ADNumberType>>>
          lqph = quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          lqph[q_point]->setup_lqp(parameters, time);
      }
  }

  // Solve the non-linear system using a Newton-Raphson scheme
  template <int dim>
  void
  Solid<dim>::solve_nonlinear_timestep(
    TrilinosWrappers::MPI::BlockVector &solution_delta_OUT)
  {
    // double start = MPI_Wtime();

    // Print the load step
    pcout << std::endl
          << "\nTimestep " << time->get_timestep() << " @ "
          << time->get_current() << "s" << std::endl;
    outfile << std::endl
            << "\nTimestep " << time->get_timestep() << " @ "
            << time->get_current() << "s" << std::endl;

    // Declare newton_update vector (solution of a Newton iteration),
    // which must have as many positions as global DoFs.
    TrilinosWrappers::MPI::BlockVector newton_update(locally_owned_partitioning,
                                                     mpi_communicator);

    // Reset the error storage objects
    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();
    error_residual_norm_last.reset();
    error_update_norm_last.reset();

    print_conv_header();

    // Declare and initialize iterator for the Newton-Raphson algorithm steps
    unsigned int newton_iteration = 0;

    // Iterate until error is below tolerance or max number iterations are
    // reached
    while (newton_iteration < parameters.max_iterations_NR)
      {
        pcout << " " << std::setw(2) << newton_iteration << " " << std::flush;
        outfile << " " << std::setw(2) << newton_iteration << " " << std::flush;

        // Initialize global stiffness matrix and global force vector to zero
        tangent_matrix = 0.0;
        system_rhs     = 0.0;

        tangent_matrix_nb = 0.0;
        system_rhs_nb     = 0.0;

        // Apply boundary conditions
        make_constraints(newton_iteration);
        assemble_system(solution_delta_OUT);

        // Compute the rhs residual (error between external and internal forces
        // in FE system)
        get_error_residual(error_residual);

        // error_residual in first iteration is stored to normalize posterior
        // error measures
        if (newton_iteration == 0)
          error_residual_0 = error_residual;

        // Determine the normalised residual error
        error_residual_norm = error_residual;
        error_residual_norm.normalise(error_residual_0);

        // If both errors are below the tolerances, exit the loop.
        //  We need to check the residual vector directly for convergence
        //  in the load steps where no external forces or displacements are
        //  imposed.
        if (((newton_iteration > 0) &&
             (error_update_norm.u <= parameters.tol_u) &&
             (error_update_norm.p_fluid <= parameters.tol_p_fluid) &&
             (error_residual_norm.u <= parameters.tol_f) &&
             (error_residual_norm.p_fluid <= parameters.tol_f)) ||
            ((newton_iteration > 0) &&
             system_rhs.l2_norm() <= parameters.tol_f))
          {
            pcout << "\n ***** CONVERGED! *****     " << system_rhs.l2_norm()
                  << "      "
                  << "  " << error_residual_norm.norm << "  "
                  << error_residual_norm.u << "  "
                  << error_residual_norm.p_fluid << "        "
                  << error_update_norm.norm << "  " << error_update_norm.u
                  << "  " << error_update_norm.p_fluid << "  " << std::endl;
            outfile << "\n ***** CONVERGED! *****     " << system_rhs.l2_norm()
                    << "      "
                    << "  " << error_residual_norm.norm << "  "
                    << error_residual_norm.u << "  "
                    << error_residual_norm.p_fluid << "        "
                    << error_update_norm.norm << "  " << error_update_norm.u
                    << "  " << error_update_norm.p_fluid << "  " << std::endl;
            print_conv_footer();

            break;
          }

        //            if (newton_iteration > 0 && error_residual_norm.norm >
        //            error_residual_norm_last.norm) { 	break; } else {
        //            	error_residual_norm_last = error_residual_norm;
        //            }

        // Solve the linearized system
        solve_linear_system(newton_update);
        constraints.distribute(newton_update);

        // Compute the displacement error
        get_error_update(newton_update, error_update);

        // error_update in first iteration is stored to normalize posterior
        // error measures
        if (newton_iteration == 0)
          error_update_0 = error_update;

        // Determine the normalised Newton update error
        error_update_norm = error_update;
        error_update_norm.normalise(error_update_0);

        // Determine the normalised residual error
        error_residual_norm = error_residual;
        error_residual_norm.normalise(error_residual_0);

        // Print error values
        pcout << " |   " << std::fixed << std::setprecision(3) << std::setw(7)
              << std::scientific << system_rhs.l2_norm() << "        "
              << error_residual_norm.norm << "  " << error_residual_norm.u
              << "  " << error_residual_norm.p_fluid << "        "
              << error_update_norm.norm << "  " << error_update_norm.u << "  "
              << error_update_norm.p_fluid << "  " << std::endl;

        outfile << " |   " << std::fixed << std::setprecision(3) << std::setw(7)
                << std::scientific << system_rhs.l2_norm() << "        "
                << error_residual_norm.norm << "  " << error_residual_norm.u
                << "  " << error_residual_norm.p_fluid << "        "
                << error_update_norm.norm << "  " << error_update_norm.u << "  "
                << error_update_norm.p_fluid << "  " << std::endl;

        // Update
        solution_delta_OUT += newton_update;
        newton_update = 0.0;
        newton_iteration++;
      }

    // If maximum allowed number of iterations for Newton algorithm are reached,
    // print non-convergence message and abort program
    AssertThrow(newton_iteration < parameters.max_iterations_NR,
                ExcMessage("No convergence in nonlinear solver!"));

    // double end = MPI_Wtime();

    /*if (this_mpi_process == 0) {
      std::ofstream solve_nonlinear_timestep_time;
      solve_nonlinear_timestep_time.open(parameters.output_directory +
    "/solve_nonlinear_timestep_time", std::ofstream::app);
      solve_nonlinear_timestep_time << std::setprecision(6) << std::scientific;
      solve_nonlinear_timestep_time << std::setw(16) <<
    this->time->get_current() << ","
          << std::setw(16) << end - start << std::endl;
      solve_nonlinear_timestep_time.close();
    }*/
  }

  // Prints the header for convergence info on console
  template <int dim>
  void
  Solid<dim>::print_conv_header()
  {
    static const unsigned int l_width = 120;

    for (unsigned int i = 0; i < l_width; ++i)
      {
        pcout << "_";
        outfile << "_";
      }

    pcout << std::endl;
    outfile << std::endl;

    pcout << "\n       SOLVER STEP      |    SYS_RES         "
          << "RES_NORM     RES_U      RES_P           "
          << "NU_NORM     NU_U       NU_P " << std::endl;
    outfile << "\n       SOLVER STEP      |    SYS_RES         "
            << "RES_NORM     RES_U      RES_P           "
            << "NU_NORM     NU_U       NU_P " << std::endl;

    for (unsigned int i = 0; i < l_width; ++i)
      {
        pcout << "_";
        outfile << "_";
      }
    pcout << std::endl << std::endl;
    outfile << std::endl << std::endl;
  }

  // Prints the footer for convergence info on console
  template <int dim>
  void
  Solid<dim>::print_conv_footer()
  {
    static const unsigned int l_width = 120;

    for (unsigned int i = 0; i < l_width; ++i)
      {
        pcout << "_";
        outfile << "_";
      }
    pcout << std::endl << std::endl;
    outfile << std::endl << std::endl;

    pcout << "Relative errors:" << std::endl
          << "Displacement:  " << error_update.u / error_update_0.u << std::endl
          << "Force (displ): " << error_residual.u / error_residual_0.u
          << std::endl
          << "Pore pressure: " << error_update.p_fluid / error_update_0.p_fluid
          << std::endl
          << "Force (pore):  "
          << error_residual.p_fluid / error_residual_0.p_fluid << std::endl;
    outfile << "Relative errors:" << std::endl
            << "Displacement:  " << error_update.u / error_update_0.u
            << std::endl
            << "Force (displ): " << error_residual.u / error_residual_0.u
            << std::endl
            << "Pore pressure: "
            << error_update.p_fluid / error_update_0.p_fluid << std::endl
            << "Force (pore):  "
            << error_residual.p_fluid / error_residual_0.p_fluid << std::endl;
  }

  // Determine the true residual error for the problem
  template <int dim>
  void
  Solid<dim>::get_error_residual(Errors &error_residual_OUT)
  {
    TrilinosWrappers::MPI::BlockVector error_res(system_rhs);
    constraints.set_zero(error_res);

    error_residual_OUT.norm    = error_res.l2_norm();
    error_residual_OUT.u       = error_res.block(u_block).l2_norm();
    error_residual_OUT.p_fluid = error_res.block(p_fluid_block).l2_norm();
  }

  // Determine the true Newton update error for the problem
  template <int dim>
  void
  Solid<dim>::get_error_update(
    const TrilinosWrappers::MPI::BlockVector &newton_update_IN,
    Errors                                   &error_update_OUT)
  {
    TrilinosWrappers::MPI::BlockVector error_ud(newton_update_IN);
    constraints.set_zero(error_ud);

    error_update_OUT.norm    = error_ud.l2_norm();
    error_update_OUT.u       = error_ud.block(u_block).l2_norm();
    error_update_OUT.p_fluid = error_ud.block(p_fluid_block).l2_norm();
  }

  // Compute the total solution, which is valid at any Newton step. This is
  // required as, to reduce computational error, the total solution is only
  // updated at the end of the timestep.
  template <int dim>
  TrilinosWrappers::MPI::BlockVector
  Solid<dim>::get_total_solution(
    const TrilinosWrappers::MPI::BlockVector &solution_delta_IN) const
  {
    // Cell interpolation -> Ghosted vector
    TrilinosWrappers::MPI::BlockVector solution_total(
      locally_owned_partitioning,
      locally_relevant_partitioning,
      mpi_communicator,
      /*vector_writable = */ false);
    TrilinosWrappers::MPI::BlockVector tmp(solution_total);
    solution_total = solution_n;
    tmp            = solution_delta_IN;
    solution_total += tmp;
    return solution_total;
  }

  // Compute elemental stiffness tensor and right-hand side force vector, and
  // assemble into global ones
  template <int dim>
  void
  Solid<dim>::assemble_system(
    const TrilinosWrappers::MPI::BlockVector &solution_delta)
  {
    double start = MPI_Wtime();

    timerconsole.enter_subsection("Assemble system");
    timerfile.enter_subsection("Assemble system");
    pcout << " ASM_SYS " << std::flush;
    outfile << " ASM_SYS " << std::flush;

    const TrilinosWrappers::MPI::BlockVector solution_total(
      get_total_solution(solution_delta));

    // Info given to FEValues and FEFaceValues constructors, to indicate which
    // data will be needed at each element.
    const UpdateFlags uf_cell(update_values | update_gradients |
                              update_quadrature_points | // may be removed again
                              update_JxW_values);

    const UpdateFlags uf_face(update_values | update_gradients |
                              update_normal_vectors | update_quadrature_points |
                              update_JxW_values);

    // Setup a copy of the data structures required for the process and pass
    // them, along with the memory addresses of the assembly functions to the
    // WorkStream object for processing
    PerTaskData_ASM per_task_data(dofs_per_cell);

    ScratchData_ASM<ADNumberType> scratch_data(
      fe, qf_cell, uf_cell, qf_face, uf_face, solution_total);

    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator> cell(
      IteratorFilters::LocallyOwnedCell(), dof_handler_ref.begin_active()),
      endc(IteratorFilters::LocallyOwnedCell(), dof_handler_ref.end());

    for (; cell != endc; ++cell)
      {
        Assert(cell->is_locally_owned(), ExcInternalError());
        Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

        assemble_system_one_cell(cell, scratch_data, per_task_data);
        copy_local_to_global_system(per_task_data);
      }

    tangent_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    tangent_matrix_nb.compress(VectorOperation::add);
    system_rhs_nb.compress(VectorOperation::add);

    timerconsole.leave_subsection();
    timerfile.leave_subsection();
    double end = MPI_Wtime();

    if (this_mpi_process == 0)
      {
        std::ofstream assemble_system_time;
        assemble_system_time.open(parameters.output_directory +
                                    "/assemble_system_time",
                                  std::ofstream::app);
        assemble_system_time << std::setprecision(6) << std::scientific;
        assemble_system_time << std::setw(16) << this->time->get_current()
                             << "," << std::setw(16) << end - start
                             << std::endl;
        assemble_system_time.close();
      }
  }

  // Add the local elemental contribution to the global stiffness tensor
  //  We do it twice, for the block and the non-block systems
  template <int dim>
  void
  Solid<dim>::copy_local_to_global_system(const PerTaskData_ASM &data)
  {
    constraints.distribute_local_to_global(data.cell_matrix,
                                           data.cell_rhs,
                                           data.local_dof_indices,
                                           tangent_matrix,
                                           system_rhs);

    constraints.distribute_local_to_global(data.cell_matrix,
                                           data.cell_rhs,
                                           data.local_dof_indices,
                                           tangent_matrix_nb,
                                           system_rhs_nb);
  }

  // Compute stiffness matrix and corresponding rhs for one element
  template <int dim>
  void
  Solid<dim>::assemble_system_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_ASM<ADNumberType>                        &scratch,
    PerTaskData_ASM                                      &data) const
  {
    Assert(cell->is_locally_owned(), ExcInternalError());

    data.reset();
    scratch.reset();
    scratch.fe_values_ref.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    // Setup automatic differentiation
    for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        // Initialise the dofs for the cell using the current solution.
        scratch.local_dof_values[k] =
          scratch.solution_total[data.local_dof_indices[k]];
        // Mark this cell DoF as an independent variable
        scratch.local_dof_values[k].diff(k, dofs_per_cell);
      }

    // Update the quadrature point solution
    // Compute the values and gradients of the solution in terms of the AD
    // variables
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            const unsigned int k_group = fe.system_to_base_index(k).first.first;
            if (k_group == u_block)
              {
                const Tensor<2, dim> Grad_Nx_u =
                  scratch.fe_values_ref[u_fe].gradient(k, q);
                for (unsigned int dd = 0; dd < dim; dd++)
                  {
                    for (unsigned int ee = 0; ee < dim; ee++)
                      {
                        scratch.solution_grads_u_total[q][dd][ee] +=
                          scratch.local_dof_values[k] * Grad_Nx_u[dd][ee];
                      }
                  }
              }
            else if (k_group == p_fluid_block)
              {
                const double Nx_p =
                  scratch.fe_values_ref[p_fluid_fe].value(k, q);
                const Tensor<1, dim> Grad_Nx_p =
                  scratch.fe_values_ref[p_fluid_fe].gradient(k, q);

                scratch.solution_values_p_fluid_total[q] +=
                  scratch.local_dof_values[k] * Nx_p;
                for (unsigned int dd = 0; dd < dim; dd++)
                  {
                    scratch.solution_grads_p_fluid_total[q][dd] +=
                      scratch.local_dof_values[k] * Grad_Nx_p[dd];
                  }
              }
            else
              Assert(k_group <= p_fluid_block, ExcInternalError());
          }
      }

    // Set up pointer "lgph" to the PointHistory object of this element
    const std::vector<std::shared_ptr<const PointHistory<dim, ADNumberType>>>
      lqph = quadrature_point_history.get_data(cell);
    Assert(lqph.size() == n_q_points, ExcInternalError());


    // Precalculate the element shape function values and gradients
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        Tensor<2, dim, ADNumberType> F_AD =
          scratch.solution_grads_u_total[q_point];
        F_AD +=
          Tensor<2, dim, double>(Physics::Elasticity::StandardTensors<dim>::I);

        double det_F_AG = Tensor<0, dim, double>(determinant(F_AD));
        double n_0S     = this->parameters.solid_vol_frac;
        Assert(det_F_AG > n_0S, ExcInternalError());
        (void)det_F_AG;
        (void)n_0S;

        Assert(determinant(F_AD) > 0, ExcMessage("Invalid deformation map"));
        const Tensor<2, dim, ADNumberType> F_inv_AD = invert(F_AD);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int i_group = fe.system_to_base_index(i).first.first;

            if (i_group == u_block)
              {
                scratch.Nx[q_point][i] =
                  scratch.fe_values_ref[u_fe].value(i, q_point);
                scratch.grad_Nx[q_point][i] =
                  scratch.fe_values_ref[u_fe].gradient(i, q_point) * F_inv_AD;
                scratch.symm_grad_Nx[q_point][i] =
                  symmetrize(scratch.grad_Nx[q_point][i]);
              }
            else if (i_group == p_fluid_block)
              {
                scratch.Nx_p_fluid[q_point][i] =
                  scratch.fe_values_ref[p_fluid_fe].value(i, q_point);
                scratch.grad_Nx_p_fluid[q_point][i] =
                  scratch.fe_values_ref[p_fluid_fe].gradient(i, q_point) *
                  F_inv_AD;
              }
            else
              Assert(i_group <= p_fluid_block, ExcInternalError());
          }
      }

    // Assemble the stiffness matrix and rhs vector
    std::vector<ADNumberType> residual_ad(dofs_per_cell, ADNumberType(0.0));
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        Tensor<2, dim, ADNumberType> F_AD =
          scratch.solution_grads_u_total[q_point];
        F_AD +=
          Tensor<2, dim, double>(Physics::Elasticity::StandardTensors<dim>::I);
        const ADNumberType det_F_AD = determinant(F_AD);

        Assert(det_F_AD > 0, ExcInternalError());
        const Tensor<2, dim, ADNumberType> F_inv_AD =
          invert(F_AD); // inverse of def. gradient tensor

        const ADNumberType p_fluid =
          scratch.solution_values_p_fluid_total[q_point];

        {
          PointHistory<dim, ADNumberType> *lqph_q_point_nc =
            const_cast<PointHistory<dim, ADNumberType> *>(lqph[q_point].get());
          lqph_q_point_nc->update_internal_equilibrium(F_AD);
        }

        // Get some info from constitutive model of solid
        static const SymmetricTensor<2, dim, double> I(
          Physics::Elasticity::StandardTensors<dim>::I);
        const SymmetricTensor<2, dim, ADNumberType> tau_E =
          lqph[q_point]->get_tau_E(F_AD);
        SymmetricTensor<2, dim, ADNumberType> tau_fluid_vol(I);
        tau_fluid_vol *= -1.0 * p_fluid * det_F_AD;


        if (parameters.eigenvalue_analysis)
          {
            const Point<dim> q_point_coord =
              scratch.fe_values_ref.quadrature_point(q_point);

            // Deformation gradient
            double         det_F_AG = Tensor<0, dim, double>(determinant(F_AD));
            Tensor<2, dim> F_AG     = Tensor<2, dim, double>(F_AD);

            // Print deformation gradient to file
            std::ofstream F;
            F.open(parameters.output_directory + "/F", std::ofstream::app);
            F << std::setprecision(6) << std::scientific;
            F << std::setw(16) << this->time->get_current() << ","
              << std::setw(16) << cell->id() << "," << std::setw(16) << q_point
              << "," << std::setw(16) << q_point_coord[0] << ","
              << std::setw(16) << q_point_coord[1] << "," << std::setw(16)
              << q_point_coord[2] << "," << std::setw(16) << F_AG[0][0] << ","
              << std::setw(16) << F_AG[0][1] << "," << std::setw(16)
              << F_AG[0][2] << "," << std::setw(16) << F_AG[1][0] << ","
              << std::setw(16) << F_AG[1][1] << "," << std::setw(16)
              << F_AG[1][2] << "," << std::setw(16) << F_AG[2][0] << ","
              << std::setw(16) << F_AG[2][1] << "," << std::setw(16)
              << F_AG[2][2] << "," << std::endl;
            F.close();

            // Print Jacobian to file
            std::ofstream det_F;
            det_F.open(parameters.output_directory + "/det_F",
                       std::ofstream::app);
            det_F << std::setprecision(6) << std::scientific;
            det_F << std::setw(16) << this->time->get_current() << ","
                  << std::setw(16) << cell->id() << "," << std::setw(16)
                  << q_point << "," << std::setw(16) << q_point_coord[0] << ","
                  << std::setw(16) << q_point_coord[1] << "," << std::setw(16)
                  << q_point_coord[2] << "," << std::setw(16) << det_F_AG << ","
                  << std::endl;
            det_F.close();


            // Left Cauchy-Green tensor
            const SymmetricTensor<2, dim, ADNumberType> B =
              symmetrize(F_AD * transpose(F_AD));

            const std::
              array<std::pair<ADNumberType, Tensor<1, dim, ADNumberType>>, dim>
                eigen_B = eigenvectors(
                  B, SymmetricTensorEigenvectorMethod::ql_implicit_shifts);

            double lambda_B_1     = Tensor<0, dim, double>(eigen_B[0].first);
            double lambda_B_2     = Tensor<0, dim, double>(eigen_B[1].first);
            double lambda_B_3     = Tensor<0, dim, double>(eigen_B[2].first);
            Tensor<1, dim> ev_B_1 = Tensor<1, dim, double>(eigen_B[0].second);
            Tensor<1, dim> ev_B_2 = Tensor<1, dim, double>(eigen_B[1].second);
            Tensor<1, dim> ev_B_3 = Tensor<1, dim, double>(eigen_B[2].second);

            // Print eigenvalues of left Cauchy-Green tensor to file
            std::ofstream eigenvalues_B;
            eigenvalues_B.open(parameters.output_directory + "/eigenvalues_B",
                               std::ofstream::app);
            eigenvalues_B << std::setprecision(6) << std::scientific;
            eigenvalues_B << std::setw(16) << this->time->get_current() << ","
                          << std::setw(16) << cell->id() << "," << std::setw(16)
                          << q_point << "," << std::setw(16) << q_point_coord[0]
                          << "," << std::setw(16) << q_point_coord[1] << ","
                          << std::setw(16) << q_point_coord[2] << ","
                          << std::setw(16) << lambda_B_1 << "," << std::setw(16)
                          << lambda_B_2 << "," << std::setw(16) << lambda_B_3
                          << std::endl;
            eigenvalues_B.close();

            // Print eigenvectors of left Cauchy-Green tensor to file
            std::ofstream eigenvectors_B;
            eigenvectors_B.open(parameters.output_directory + "/eigenvectors_B",
                                std::ofstream::app);
            eigenvectors_B << std::setprecision(6) << std::scientific;
            eigenvectors_B << std::setw(16) << this->time->get_current() << ","
                           << std::setw(16) << cell->id() << ","
                           << std::setw(16) << q_point << "," << std::setw(16)
                           << q_point_coord[0] << "," << std::setw(16)
                           << q_point_coord[1] << "," << std::setw(16)
                           << q_point_coord[2] << "," << std::setw(16)
                           << ev_B_1[0] << "," << std::setw(16) << ev_B_1[1]
                           << "," << std::setw(16) << ev_B_1[2] << ","
                           << std::setw(16) << ev_B_2[0] << "," << std::setw(16)
                           << ev_B_2[1] << "," << std::setw(16) << ev_B_2[2]
                           << "," << std::setw(16) << ev_B_3[0] << ","
                           << std::setw(16) << ev_B_3[1] << "," << std::setw(16)
                           << ev_B_3[2] << std::endl;
            eigenvectors_B.close();

            // Cauchy stress
            const SymmetricTensor<2, dim, ADNumberType> cauchy_E_base_AD =
              lqph[q_point]->get_Cauchy_E_base(F_AD);
            const SymmetricTensor<2, dim, ADNumberType> cauchy_E_ext_func_AD =
              lqph[q_point]->get_Cauchy_E_ext_func(F_AD);
            const SymmetricTensor<2, dim> cauchy_E_base =
              SymmetricTensor<2, dim, double>(cauchy_E_base_AD);
            const SymmetricTensor<2, dim> cauchy_E_ext_func =
              SymmetricTensor<2, dim, double>(cauchy_E_ext_func_AD);

            // Print isochoric Cauchy stress to file
            std::ofstream cauchy_iso;
            cauchy_iso.open(parameters.output_directory + "/cauchy_iso",
                            std::ofstream::app);
            cauchy_iso << std::setprecision(6) << std::scientific;
            cauchy_iso << std::setw(16) << this->time->get_current() << ","
                       << std::setw(16) << cell->id() << "," << std::setw(16)
                       << q_point << "," << std::setw(16) << q_point_coord[0]
                       << "," << std::setw(16) << q_point_coord[1] << ","
                       << std::setw(16) << q_point_coord[2] << ","
                       << std::setw(16) << cauchy_E_base[0][0] << ","
                       << std::setw(16) << cauchy_E_base[0][1] << ","
                       << std::setw(16) << cauchy_E_base[0][2] << ","
                       << std::setw(16) << cauchy_E_base[1][0] << ","
                       << std::setw(16) << cauchy_E_base[1][1] << ","
                       << std::setw(16) << cauchy_E_base[1][2] << ","
                       << std::setw(16) << cauchy_E_base[2][0] << ","
                       << std::setw(16) << cauchy_E_base[2][1] << ","
                       << std::setw(16) << cauchy_E_base[2][2] << ","
                       << std::endl;
            cauchy_iso.close();

            // Print volumetric Cauchy stress to file
            std::ofstream cauchy_vol;
            cauchy_vol.open(parameters.output_directory + "/cauchy_vol",
                            std::ofstream::app);
            cauchy_vol << std::setprecision(6) << std::scientific;
            cauchy_vol << std::setw(16) << this->time->get_current() << ","
                       << std::setw(16) << cell->id() << "," << std::setw(16)
                       << q_point << "," << std::setw(16) << q_point_coord[0]
                       << "," << std::setw(16) << q_point_coord[1] << ","
                       << std::setw(16) << q_point_coord[2] << ","
                       << std::setw(16) << cauchy_E_ext_func[0][0] << ","
                       << std::setw(16) << cauchy_E_ext_func[0][1] << ","
                       << std::setw(16) << cauchy_E_ext_func[0][2] << ","
                       << std::setw(16) << cauchy_E_ext_func[1][0] << ","
                       << std::setw(16) << cauchy_E_ext_func[1][1] << ","
                       << std::setw(16) << cauchy_E_ext_func[1][2] << ","
                       << std::setw(16) << cauchy_E_ext_func[2][0] << ","
                       << std::setw(16) << cauchy_E_ext_func[2][1] << ","
                       << std::setw(16) << cauchy_E_ext_func[2][2] << ","
                       << std::endl;
            cauchy_vol.close();
          }

        // Get some info from constitutive model of fluid
        const ADNumberType det_F_aux = lqph[q_point]->get_converged_det_F();
        const double       det_F_converged = Tensor<0, dim, double>(
          det_F_aux); // Needs to be double, not AD number
        const Tensor<1, dim, ADNumberType> overall_body_force =
          lqph[q_point]->get_overall_body_force(F_AD, parameters);

        // Define some aliases to make the assembly process easier to follow
        const std::vector<Tensor<1, dim>> &Nu = scratch.Nx[q_point];
        const std::vector<SymmetricTensor<2, dim, ADNumberType>> &symm_grad_Nu =
          scratch.symm_grad_Nx[q_point];
        const std::vector<double> &Np = scratch.Nx_p_fluid[q_point];
        const std::vector<Tensor<1, dim, ADNumberType>> &grad_Np =
          scratch.grad_Nx_p_fluid[q_point];
        const Tensor<1, dim, ADNumberType> grad_p =
          scratch.solution_grads_p_fluid_total[q_point] * F_inv_AD;
        const double JxW = scratch.fe_values_ref.JxW(q_point);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int i_group = fe.system_to_base_index(i).first.first;

            if (i_group == u_block)
              {
                residual_ad[i] +=
                  symm_grad_Nu[i] * (tau_E + tau_fluid_vol) * JxW;
                residual_ad[i] -= Nu[i] * overall_body_force * JxW;
              }
            else if (i_group == p_fluid_block)
              {
                const Tensor<1, dim, ADNumberType> seepage_vel_current =
                  lqph[q_point]->get_seepage_velocity_current(F_AD, grad_p);
                residual_ad[i] += Np[i] * (det_F_AD - det_F_converged) * JxW;
                residual_ad[i] -=
                  time->get_delta_t() * grad_Np[i] * seepage_vel_current * JxW;
              }
            else
              Assert(i_group <= p_fluid_block, ExcInternalError());
          }
      }

    // Assemble the Neumann contribution (external force contribution).
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
         ++face) // Loop over faces in element
      {
        if (cell->face(face)->at_boundary() == true)
          {
            scratch.fe_face_values_ref.reinit(cell, face);

            for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                 ++f_q_point)
              {
                const Tensor<1, dim> &N =
                  scratch.fe_face_values_ref.normal_vector(f_q_point);
                const Point<dim> &pt =
                  scratch.fe_face_values_ref.quadrature_point(f_q_point);
                const Tensor<1, dim> traction =
                  get_neumann_traction(cell->face(face)->boundary_id(), pt, N);
                const double flow =
                  get_prescribed_fluid_flow(cell->face(face)->boundary_id(),
                                            pt);

                if ((traction.norm() < 1e-12) && (std::abs(flow) < 1e-12))
                  continue;

                const double JxW_f = scratch.fe_face_values_ref.JxW(f_q_point);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const unsigned int i_group =
                      fe.system_to_base_index(i).first.first;

                    if ((i_group == u_block) && (traction.norm() > 1e-12))
                      {
                        const unsigned int component_i =
                          fe.system_to_component_index(i).first;
                        const double Nu_f =
                          scratch.fe_face_values_ref.shape_value(i, f_q_point);
                        residual_ad[i] -=
                          (Nu_f * traction[component_i]) * JxW_f;
                      }
                    if ((i_group == p_fluid_block) && (std::abs(flow) > 1e-12))
                      {
                        const double Nu_p =
                          scratch.fe_face_values_ref.shape_value(i, f_q_point);
                        residual_ad[i] -= (Nu_p * flow) * JxW_f;
                      }
                  }
              }
          }
      }

    // Linearise the residual
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const ADNumberType &R_i = residual_ad[i];

        data.cell_rhs(i) -= R_i.val();
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          data.cell_matrix(i, j) += R_i.fastAccessDx(j);
      }
  }

  // Store the converged values of the internal variables
  // template <int dim> void Solid<dim>::update_end_timestep()
  template <int dim>
  double
  Solid<dim>::update_end_timestep()
  {
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator> cell(
      IteratorFilters::LocallyOwnedCell(), dof_handler_ref.begin_active()),
      endc(IteratorFilters::LocallyOwnedCell(), dof_handler_ref.end());

    std::vector<double> det_F;
    std::vector<double> det_F_old;
    double              det_F_min;
    double              det_F_min_old;
    double              det_F_change;

    for (; cell != endc; ++cell)
      {
        Assert(cell->is_locally_owned(), ExcInternalError());
        Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

        const std::vector<std::shared_ptr<PointHistory<dim, ADNumberType>>>
          lqph = quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            det_F_old.push_back(lqph[q_point]->get_converged_det_F());
            lqph[q_point]->update_end_timestep();
            det_F.push_back(lqph[q_point]->get_converged_det_F());
          }
      }
    det_F_min_old       = *std::min_element(det_F_old.begin(), det_F_old.end());
    det_F_min           = *std::min_element(det_F.begin(), det_F.end());
    return det_F_change = (det_F_min_old - det_F_min) /
                          (det_F_min_old - parameters.solid_vol_frac);
  }

  // Compute the change of the Jacobian on the faces to adjust the timestep
  template <int dim>
  double
  Solid<dim>::jacobian_on_faces(TrilinosWrappers::MPI::BlockVector solution_IN)
  {
    TrilinosWrappers::MPI::BlockVector solution_total(
      locally_owned_partitioning,
      locally_relevant_partitioning,
      mpi_communicator,
      false);
    solution_total = solution_IN;

    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator> cell(
      IteratorFilters::LocallyOwnedCell(), dof_handler_ref.begin_active()),
      endc(IteratorFilters::LocallyOwnedCell(), dof_handler_ref.end());

    std::vector<double> det_F;
    double              det_F_min;
    static double       det_F_min_old = 1;
    double              det_F_change;

    for (; cell != endc; ++cell)
      {
        Assert(cell->is_locally_owned(), ExcInternalError());
        Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

        const UpdateFlags uf_face(update_gradients);
        FEFaceValues<dim> fe_face_values_ref(mapping, fe, qf_face, uf_face);

        // start face loop
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            // check if face is at the boundary
            if (cell->face(face)->at_boundary() == true)
              {
                fe_face_values_ref.reinit(cell, face);

                // Get displacement gradients for current face
                std::vector<Tensor<2, dim>> solution_grads_u_f(n_q_points_f);
                fe_face_values_ref[u_fe].get_function_gradients(
                  solution_total, solution_grads_u_f);

                // start Gauss points loop on faces
                for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                     ++f_q_point)
                  {
                    // Compute deformation gradient from displacements gradient
                    // and its Jacobian
                    const Tensor<2, dim, ADNumberType> F_AD =
                      Physics::Elasticity::Kinematics::F(
                        solution_grads_u_f[f_q_point]);
                    ADNumberType det_F_AD = determinant(F_AD);
                    det_F.push_back(Tensor<0, dim, double>(det_F_AD));
                  }
              }
          }
      }
    det_F_min = *std::min_element(det_F.begin(), det_F.end());
    det_F_change =
      (det_F_min_old - det_F_min) / (det_F_min_old - parameters.solid_vol_frac);
    det_F_min_old = det_F_min;
    return det_F_change;
  }


  // Solve the linearized equations
  template <int dim>
  void
  Solid<dim>::solve_linear_system(
    TrilinosWrappers::MPI::BlockVector &newton_update_OUT)
  {
    timerconsole.enter_subsection("Linear solver");
    timerfile.enter_subsection("Linear solver");
    pcout << " SLV " << std::flush;
    outfile << " SLV " << std::flush;

    TrilinosWrappers::MPI::Vector newton_update_nb;
    newton_update_nb.reinit(locally_owned_dofs, mpi_communicator);

    SolverControl solver_control(
      tangent_matrix_nb.m(), // (maximum number of iterations, tolerance)
      1.0e-8 * system_rhs_nb.l2_norm());
    TrilinosWrappers::SolverDirect::AdditionalData
      additional_data; // select solver type
    additional_data.solver_type =
      "Amesos_Superludist"; // default: Amesos_Klu Superludist
    TrilinosWrappers::SolverDirect solver(solver_control, additional_data);

    double start = MPI_Wtime();
    solver.solve(tangent_matrix_nb,
                 newton_update_nb,
                 system_rhs_nb); // linear system (A, x, b)
    double end = MPI_Wtime();

    // print linear system
    //           std::ofstream tangent_matrix;
    //           tangent_matrix.open("tangent_matrix", std::ofstream::app);
    //		   tangent_matrix_nb.print(tangent_matrix);
    //		   tangent_matrix.close();

    // Copy the non-block solution back to block system
    for (unsigned int i = 0; i < locally_owned_dofs.n_elements(); ++i)
      {
        const types::global_dof_index idx_i =
          locally_owned_dofs.nth_index_in_set(i);
        newton_update_OUT(idx_i) = newton_update_nb(idx_i);
      }
    newton_update_OUT.compress(VectorOperation::insert);

    timerconsole.leave_subsection();
    timerfile.leave_subsection();

    if (this_mpi_process == 0)
      {
        std::ofstream solve_linear_system_time;
        solve_linear_system_time.open(parameters.output_directory +
                                        "/solve_linear_system_time",
                                      std::ofstream::app);
        solve_linear_system_time << std::setprecision(6) << std::scientific;
        solve_linear_system_time << std::setw(16) << this->time->get_current()
                                 << "," << std::setw(16) << end - start
                                 << std::endl;
        solve_linear_system_time.close();
      }
  }


  // Class to compute gradient of the pressure
  template <int dim>
  class PressGradPostproc : public DataPostprocessorVector<dim>
  {
  public:
    PressGradPostproc(const unsigned int p_fluid_component)
      : DataPostprocessorVector<dim>("pressure_gradient", update_gradients)
      , p_fluid_component(p_fluid_component)
    {}

    virtual ~PressGradPostproc()
    {}

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      const unsigned int max_points = input_data.solution_gradients.size();
      for (unsigned int p = 0; p < max_points; ++p)
        {
          AssertDimension(computed_quantities[p].size(), dim);
          for (unsigned int d = 0; d < dim; ++d)
            computed_quantities[p][d] =
              input_data.solution_gradients[p][p_fluid_component][d];
        }
    }

  private:
    const unsigned int p_fluid_component;
  };

  // Class to compute stresses
  template <int dim>
  class CauchyStressesPostproc : public DataPostprocessor<dim>
  {
  public:
    CauchyStressesPostproc(const Parameters::AllParameters &parameters,
                           const std::shared_ptr<Time>      time)
      : parameters(parameters)
      , time(time)
    {}

    virtual ~CauchyStressesPostproc()
    {}

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      const unsigned int max_points = input_data.solution_values.size();
      for (unsigned int p = 0; p < max_points; ++p)
        {
          // Compute deformation gradient tensor of the point
          Tensor<2, dim> grad_u;
          for (unsigned int d = 0; d < dim; ++d)
            grad_u[d] = input_data.solution_gradients[p][d];

          const Tensor<2, dim, ADNumberType> F_AD =
            Physics::Elasticity::Kinematics::F(grad_u);

          // Extract pressure value of the point
          const double p_fluid = input_data.solution_values[p][dim];

          // Compute stresses at the point
          std::shared_ptr<Material_Hyperelastic<dim, ADNumberType>>
            solid_material;

          if (parameters.mat_type == "Neo-Hooke")
            solid_material.reset(
              new NeoHooke<dim, ADNumberType>(parameters, time));
          else if (parameters.mat_type == "Neo-Hooke-PS")
            solid_material.reset(
              new NeoHookePS<dim, ADNumberType>(parameters, time));
          else if (parameters.mat_type == "Neo-Hooke-Ehlers")
            solid_material.reset(
              new NeoHookeEhlers<dim, ADNumberType>(parameters, time));
          else if (parameters.mat_type == "Ogden")
            solid_material.reset(
              new Ogden<dim, ADNumberType>(parameters, time));
          else if (parameters.mat_type == "visco-Ogden")
            solid_material.reset(
              new visco_Ogden<dim, ADNumberType>(parameters, time));
          else if (parameters.mat_type == "visco2-Ogden")
            solid_material.reset(
              new visco2_Ogden<dim, ADNumberType>(parameters, time));
          else
            Assert(false, ExcMessage("Material type not implemented"));

          const SymmetricTensor<2, dim, ADNumberType> sigma_E_AD =
            solid_material->get_Cauchy_E(F_AD);

          static const SymmetricTensor<2, dim, double> I(
            Physics::Elasticity::StandardTensors<dim>::I);

          SymmetricTensor<2, dim> sigma_E;
          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              sigma_E[i][j] = Tensor<0, dim, double>(sigma_E_AD[i][j]);

          SymmetricTensor<2, dim> sigma_fluid_vol(I);
          sigma_fluid_vol *= -p_fluid;
          const SymmetricTensor<2, dim> sigma = sigma_E + sigma_fluid_vol;

          computed_quantities[p](0) = sigma[0][0];
          computed_quantities[p](1) = sigma[1][1];
          computed_quantities[p](2) = sigma[2][2];
          computed_quantities[p](3) = sigma[0][1];
          computed_quantities[p](4) = sigma[0][2];
          computed_quantities[p](5) = sigma[1][2];

          computed_quantities[p](6)  = sigma_E[0][0];
          computed_quantities[p](7)  = sigma_E[1][1];
          computed_quantities[p](8)  = sigma_E[2][2];
          computed_quantities[p](9)  = sigma_E[0][1];
          computed_quantities[p](10) = sigma_E[0][2];
          computed_quantities[p](11) = sigma_E[1][2];

          computed_quantities[p](12) = sigma_fluid_vol[0][0];
          computed_quantities[p](13) = sigma_fluid_vol[1][1];
          computed_quantities[p](14) = sigma_fluid_vol[2][2];
          computed_quantities[p](15) = sigma_fluid_vol[0][1];
          computed_quantities[p](16) = sigma_fluid_vol[0][2];
          computed_quantities[p](17) = sigma_fluid_vol[1][2];
        }
    }

    virtual std::vector<std::string>
    get_names() const override
    {
      std::vector<std::string> solution_names;
      solution_names.emplace_back("total_cauchy_stress_xx");
      solution_names.emplace_back("total_cauchy_stress_yy");
      solution_names.emplace_back("total_cauchy_stress_zz");
      solution_names.emplace_back("total_cauchy_stress_xy");
      solution_names.emplace_back("total_cauchy_stress_xz");
      solution_names.emplace_back("total_cauchy_stress_yz");

      solution_names.emplace_back("extra_cauchy_stress_xx");
      solution_names.emplace_back("extra_cauchy_stress_yy");
      solution_names.emplace_back("extra_cauchy_stress_zz");
      solution_names.emplace_back("extra_cauchy_stress_xy");
      solution_names.emplace_back("extra_cauchy_stress_xz");
      solution_names.emplace_back("extra_cauchy_stress_yz");

      solution_names.emplace_back("volumetric_cauchy_stress_xx");
      solution_names.emplace_back("volumetric_cauchy_stress_yy");
      solution_names.emplace_back("volumetric_cauchy_stress_zz");
      solution_names.emplace_back("volumetric_cauchy_stress_xy");
      solution_names.emplace_back("volumetric_cauchy_stress_xz");
      solution_names.emplace_back("volumetric_cauchy_stress_yz");
      return solution_names;
    }

    virtual std::vector<
      DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override
    {
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(18, DataComponentInterpretation::component_is_scalar);
      return interpretation;
    }

    virtual UpdateFlags
    get_needed_update_flags() const override
    {
      return (update_values | update_gradients | update_JxW_values);
    }

  private:
    const Parameters::AllParameters parameters;
    const std::shared_ptr<Time>     time;
    using ADNumberType = Sacado::Fad::DFad<double>;
  };

  // Class to compute the seepage velocity
  template <int dim>
  class SeepageVelPostproc : public DataPostprocessorVector<dim>
  {
  public:
    SeepageVelPostproc(const Parameters::AllParameters &parameters,
                       const unsigned int               p_fluid_component)
      : DataPostprocessorVector<dim>("seepage_velocity", update_gradients)
      , parameters(parameters)
      , p_fluid_component(p_fluid_component)
    {}

    virtual ~SeepageVelPostproc()
    {}

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      const unsigned int max_points = input_data.solution_gradients.size();
      for (unsigned int p = 0; p < max_points; ++p)
        {
          // Compute deformation gradient tensor of the point
          Tensor<2, dim> grad_u;
          for (unsigned int d = 0; d < dim; ++d)
            grad_u[d] = input_data.solution_gradients[p][d];

          const Tensor<2, dim, ADNumberType> F_AD =
            Physics::Elasticity::Kinematics::F(grad_u);

          // Extract pressure gradient at the point
          Tensor<1, dim> grad_p_AD =
            input_data.solution_gradients[p][p_fluid_component];

          // Compute seepage velocity
          std::shared_ptr<Material_Darcy_Fluid<dim, ADNumberType>>
            fluid_material;
          fluid_material.reset(
            new Material_Darcy_Fluid<dim, ADNumberType>(parameters));

          const Tensor<1, dim, ADNumberType> seepage_vel_AD =
            fluid_material->get_seepage_velocity_current(F_AD, grad_p_AD);

          AssertDimension(computed_quantities[p].size(), dim);
          for (unsigned int d = 0; d < dim; ++d)
            computed_quantities[p][d] =
              Tensor<0, dim, double>(seepage_vel_AD[d]);
        }
    }

  private:
    const Parameters::AllParameters parameters;
    const unsigned int              p_fluid_component;
    using ADNumberType = Sacado::Fad::DFad<double>;
  };

  // Class to compute the Jacobian, det(F)
  template <int dim>
  class JacobianPostproc : public DataPostprocessorScalar<dim>
  {
  public:
    JacobianPostproc()
      : DataPostprocessorScalar<dim>("jacobian", update_gradients)
    {}

    virtual ~JacobianPostproc()
    {}

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      const unsigned int max_points = input_data.solution_gradients.size();
      for (unsigned int p = 0; p < max_points; ++p)
        {
          // Compute deformation gradient tensor of the point
          Tensor<2, dim> grad_u;
          for (unsigned int d = 0; d < dim; ++d)
            grad_u[d] = input_data.solution_gradients[p][d];

          const Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(grad_u);
          computed_quantities[p] = determinant(F);
        }
    }
  };

  // Class to compute the solid volume fraction, n_s
  template <int dim>
  class SolidVolFracPostproc : public DataPostprocessorScalar<dim>
  {
  public:
    SolidVolFracPostproc(double n_0s)
      : DataPostprocessorScalar<dim>("solid_volume_fraction", update_gradients)
      , n_0s(n_0s)
    {}

    virtual ~SolidVolFracPostproc()
    {}

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      const unsigned int max_points = input_data.solution_gradients.size();
      for (unsigned int p = 0; p < max_points; ++p)
        {
          // Compute deformation gradient tensor of the point
          Tensor<2, dim> grad_u;
          for (unsigned int d = 0; d < dim; ++d)
            grad_u[d] = input_data.solution_gradients[p][d];

          const Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(grad_u);
          computed_quantities[p] = n_0s / determinant(F);
        }
    }

  private:
    const double n_0s;
  };

  // Class to compute the stretches
  template <int dim>
  class StretchesPostproc : public DataPostprocessorVector<dim>
  {
  public:
    StretchesPostproc()
      : DataPostprocessorVector<dim>("stretches", update_gradients)
    {}

    virtual ~StretchesPostproc()
    {}

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      const unsigned int max_points = input_data.solution_gradients.size();
      for (unsigned int p = 0; p < max_points; ++p)
        {
          // Compute deformation gradient tensor of the point
          Tensor<2, dim> grad_u;
          for (unsigned int d = 0; d < dim; ++d)
            grad_u[d] = input_data.solution_gradients[p][d];

          const Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(grad_u);

          // Green-Lagrange strain
          const Tensor<2, dim> E_strain = Physics::Elasticity::Kinematics::E(F);

          AssertDimension(computed_quantities[p].size(), dim);
          for (unsigned int d = 0; d < dim; ++d)
            computed_quantities[p][d] = std::sqrt(1.0 + 2.0 * E_strain[d][d]);
        }
    }
  };

  // Class to compute dissipations
  template <int dim>
  class DissipPostproc : public DataPostprocessor<dim>
  {
  public:
    DissipPostproc(const Parameters::AllParameters &parameters,
                   const std::shared_ptr<Time>      time,
                   const unsigned int               p_fluid_component)
      : parameters(parameters)
      , time(time)
      , p_fluid_component(p_fluid_component)
    {}

    virtual ~DissipPostproc()
    {}

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      const unsigned int max_points = input_data.solution_values.size();
      for (unsigned int p = 0; p < max_points; ++p)
        {
          // Compute deformation gradient tensor of the point
          Tensor<2, dim> grad_u;
          for (unsigned int d = 0; d < dim; ++d)
            grad_u[d] = input_data.solution_gradients[p][d];

          const Tensor<2, dim, ADNumberType> F_AD =
            Physics::Elasticity::Kinematics::F(grad_u);

          // Extract pressure value of the point
          // const double p_fluid =  input_data.solution_values[p][dim];
          // Extract pressure gradient at the point
          Tensor<1, dim> grad_p_AD =
            input_data.solution_gradients[p][p_fluid_component];

          // Compute porous dissipation
          std::shared_ptr<Material_Darcy_Fluid<dim, ADNumberType>>
            fluid_material;
          fluid_material.reset(
            new Material_Darcy_Fluid<dim, ADNumberType>(parameters));

          const double poro_dissip =
            fluid_material->get_porous_dissipation(F_AD, grad_p_AD);

          // Compute viscous dissipation
          std::shared_ptr<Material_Hyperelastic<dim, ADNumberType>>
            solid_material;

          if (parameters.mat_type == "Neo-Hooke")
            solid_material.reset(
              new NeoHooke<dim, ADNumberType>(parameters, time));
          else if (parameters.mat_type == "Neo-Hooke-PS")
            solid_material.reset(
              new NeoHookePS<dim, ADNumberType>(parameters, time));
          else if (parameters.mat_type == "Neo-Hooke-Ehlers")
            solid_material.reset(
              new NeoHookeEhlers<dim, ADNumberType>(parameters, time));
          else if (parameters.mat_type == "Ogden")
            solid_material.reset(
              new Ogden<dim, ADNumberType>(parameters, time));
          else if (parameters.mat_type == "visco-Ogden")
            solid_material.reset(
              new visco_Ogden<dim, ADNumberType>(parameters, time));
          else if (parameters.mat_type == "visco2-Ogden")
            solid_material.reset(
              new visco2_Ogden<dim, ADNumberType>(parameters, time));
          else
            Assert(false, ExcMessage("Material type not implemented"));

          const SymmetricTensor<2, dim, ADNumberType> sigma_E_AD =
            solid_material->get_Cauchy_E(F_AD);

          const double visco_dissip = solid_material->get_viscous_dissipation();

          computed_quantities[p](0) = poro_dissip;
          computed_quantities[p](1) = visco_dissip;
        }
    }

    virtual std::vector<std::string>
    get_names() const override
    {
      std::vector<std::string> solution_names;
      solution_names.emplace_back("porous_dissipation");
      solution_names.emplace_back("viscous_dissipation");
      return solution_names;
    }

    virtual std::vector<
      DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override
    {
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(2, DataComponentInterpretation::component_is_scalar);
      return interpretation;
    }

    virtual UpdateFlags
    get_needed_update_flags() const override
    {
      return (update_values | update_gradients);
    }

  private:
    const Parameters::AllParameters parameters;
    const std::shared_ptr<Time>     time;
    const unsigned int              p_fluid_component;
    using ADNumberType = Sacado::Fad::DFad<double>;
  };


  // Print results to vtu file
  template <int dim>
  void
  Solid<dim>::output_results_to_vtu(
    const unsigned int                 timestep,
    const double                       current_time,
    TrilinosWrappers::MPI::BlockVector solution_IN) const
  {
    TrilinosWrappers::MPI::BlockVector solution_total(
      locally_owned_partitioning,
      locally_relevant_partitioning,
      mpi_communicator,
      false);
    solution_total = solution_IN;
    Vector<double> material_id;
    material_id.reinit(triangulation.n_active_cells());
    std::vector<types::subdomain_id> partition_int(
      triangulation.n_active_cells());

    // Iterate through elements (cells) to obtain material ID for each
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator> cell(
      IteratorFilters::LocallyOwnedCell(), dof_handler_ref.begin_active()),
      endc(IteratorFilters::LocallyOwnedCell(), dof_handler_ref.end());
    for (; cell != endc; ++cell)
      {
        Assert(cell->is_locally_owned(), ExcInternalError());
        Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

        material_id(cell->active_cell_index()) =
          static_cast<int>(cell->material_id());
      }

    // Add the results to the solution to create the output file for Paraview
    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      comp_type(dim, DataComponentInterpretation::component_is_part_of_vector);
    comp_type.push_back(DataComponentInterpretation::component_is_scalar);

    GridTools::get_subdomain_association(triangulation, partition_int);

    std::vector<std::string> solution_name(dim, "displacement");
    solution_name.push_back("pore_pressure");

    data_out.attach_dof_handler(dof_handler_ref);
    data_out.add_data_vector(solution_total,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             comp_type);

    const Vector<double> partitioning(partition_int.begin(),
                                      partition_int.end());

    data_out.add_data_vector(partitioning, "partitioning");
    data_out.add_data_vector(material_id, "material_id");

    PressGradPostproc<dim> pres_grad(p_fluid_component);
    data_out.add_data_vector(solution_total, pres_grad);

    CauchyStressesPostproc<dim> stresses_post(parameters, time);
    data_out.add_data_vector(solution_total, stresses_post);

    SeepageVelPostproc<dim> seepage_vel(parameters, p_fluid_component);
    data_out.add_data_vector(solution_total, seepage_vel);

    JacobianPostproc<dim> jacobian;
    data_out.add_data_vector(solution_total, jacobian);

    SolidVolFracPostproc<dim> n0s(parameters.solid_vol_frac);
    data_out.add_data_vector(solution_total, n0s);

    StretchesPostproc<dim> stretches;
    data_out.add_data_vector(solution_total, stretches);

    DissipPostproc<dim> dissipations(parameters, time, p_fluid_component);
    data_out.add_data_vector(solution_total, dissipations);

    // data_out.build_patches(degree_displ);
    data_out.build_patches(mapping,
                           degree_displ,
                           DataOut<dim>::curved_boundary);

    struct Filename
    {
      static std::string
      get_filename_vtu(unsigned int       process,
                       unsigned int       timestep,
                       const unsigned int n_digits = 5)
      {
        std::ostringstream filename_vtu;
        filename_vtu << "solution."
                     << Utilities::int_to_string(process, n_digits) << "."
                     << Utilities::int_to_string(timestep, n_digits) << ".vtu";
        return filename_vtu.str();
      }

      static std::string
      get_filename_pvtu(unsigned int timestep, const unsigned int n_digits = 5)
      {
        std::ostringstream filename_vtu;
        filename_vtu << "solution."
                     << Utilities::int_to_string(timestep, n_digits) << ".pvtu";
        return filename_vtu.str();
      }

      static std::string
      get_filename_pvd(void)
      {
        std::ostringstream filename_vtu;
        filename_vtu << "solution.pvd";
        return filename_vtu.str();
      }
    };

    const std::string filename_vtu =
      Filename::get_filename_vtu(this_mpi_process, timestep);
    std::ofstream output(parameters.output_directory + "/" +
                         filename_vtu.c_str());
    data_out.write_vtu(output);

    // We have a collection of files written in parallel
    // This next set of steps should only be performed by master process
    if (this_mpi_process == 0)
      {
        // List of all files written out at this timestep by all processors
        std::vector<std::string> parallel_filenames_vtu;
        for (unsigned int p = 0; p < n_mpi_processes; ++p)
          {
            parallel_filenames_vtu.push_back(
              Filename::get_filename_vtu(p, timestep));
          }

        const std::string filename_pvtu(Filename::get_filename_pvtu(timestep));
        std::ofstream     pvtu_master(parameters.output_directory + "/" +
                                  filename_pvtu.c_str());
        data_out.write_pvtu_record(pvtu_master, parallel_filenames_vtu);

        // Time dependent data master file
        static std::vector<std::pair<double, std::string>>
          time_and_name_history;
        time_and_name_history.push_back(
          std::make_pair(current_time, filename_pvtu));
        const std::string filename_pvd(Filename::get_filename_pvd());
        std::ofstream     pvd_output(parameters.output_directory + "/" +
                                 filename_pvd.c_str());
        DataOutBase::write_pvd_record(pvd_output, time_and_name_history);
      }
  }

  // Print boundary conditions to vtu file
  // This function os analogous to the output_results_to_vtu function,
  //  except that we only print to file the surface information.

  // NOTE: Doesn't seem to be working properly when running the code in parallel
  // Had to change "build_patches" so that output mesh is not split to match
  // displacement order (otherwise, it gives an error saying it is trying to
  // access a position in the vector that doesn't belong to this mpi process).
  // Also, solution (displ and pressure) aren't computed correctly, but load
  // seems to be.
  template <int dim>
  void
  Solid<dim>::output_bcs_to_vtu(
    const unsigned int                 timestep,
    const double                       current_time,
    TrilinosWrappers::MPI::BlockVector solution_IN) const
  {
    TrilinosWrappers::MPI::BlockVector solution_total(
      locally_owned_partitioning,
      locally_relevant_partitioning,
      mpi_communicator,
      false);
    solution_total = solution_IN;
    std::vector<types::subdomain_id> partition_int(
      triangulation.n_active_cells());

    // Declare local vectors to store values
    //  OUTPUT AVERAGED ON ELEMENTS -------------------------------------------
    std::vector<Vector<double>> loads_elements(
      dim, Vector<double>(triangulation.n_active_cells()));

    // We need to create a new FE space with a single dof per node to avoid
    // duplication of the output on nodes for our problem with dim+1 dofs.
    FE_Q<dim>       fe_vertex(1);
    DoFHandler<dim> vertex_handler_ref(triangulation);
    vertex_handler_ref.distribute_dofs(fe_vertex);
    AssertThrow(vertex_handler_ref.n_dofs() == triangulation.n_vertices(),
                ExcDimensionMismatch(vertex_handler_ref.n_dofs(),
                                     triangulation.n_vertices()));

    Vector<double> counter_on_vertices_mpi(vertex_handler_ref.n_dofs());
    Vector<double> sum_counter_on_vertices(vertex_handler_ref.n_dofs());

    Vector<double> jacobian_vertex_mpi(vertex_handler_ref.n_dofs());
    Vector<double> sum_jacobian_vertex(vertex_handler_ref.n_dofs());

    // OUTPUT AVERAGED ON NODES ----------------------------------------------
    FESystem<dim>   fe_vertex_vec(FE_Q<dim>(1), dim);
    DoFHandler<dim> vertex_vec_handler_ref(triangulation);
    vertex_vec_handler_ref.distribute_dofs(fe_vertex_vec);
    AssertThrow(vertex_vec_handler_ref.n_dofs() ==
                  (dim * triangulation.n_vertices()),
                ExcDimensionMismatch(vertex_vec_handler_ref.n_dofs(),
                                     (dim * triangulation.n_vertices())));

    Vector<double> loads_vertex_vec_mpi(vertex_vec_handler_ref.n_dofs());
    Vector<double> sum_loads_vertex_vec(vertex_vec_handler_ref.n_dofs());
    Vector<double> counter_on_vertices_vec_mpi(vertex_vec_handler_ref.n_dofs());
    Vector<double> sum_counter_on_vertices_vec(vertex_vec_handler_ref.n_dofs());

    // -----------------------------------------------------------------------

    // Declare an instance of the material class object
    if (parameters.mat_type == "Neo-Hooke")
      NeoHooke<dim, ADNumberType> material(parameters, time);
    else if (parameters.mat_type == "Neo-Hooke-PS")
      NeoHookePS<dim, ADNumberType> material(parameters, time);
    else if (parameters.mat_type == "Neo-Hooke-Ehlers")
      NeoHookeEhlers<dim, ADNumberType> material(parameters, time);
    else if (parameters.mat_type == "Ogden")
      Ogden<dim, ADNumberType> material(parameters, time);
    else if (parameters.mat_type == "visco-Ogden")
      visco_Ogden<dim, ADNumberType> material(parameters, time);
    else if (parameters.mat_type == "visco2-Ogden")
      visco2_Ogden<dim, ADNumberType> material(parameters, time);
    else
      Assert(false, ExcMessage("Material type not implemented"));

    // Iterate through elements (cells) and Gauss Points
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator> cell(
      IteratorFilters::LocallyOwnedCell(), dof_handler_ref.begin_active()),
      endc(IteratorFilters::LocallyOwnedCell(), dof_handler_ref.end()),
      cell_v(IteratorFilters::LocallyOwnedCell(),
             vertex_handler_ref.begin_active()),
      cell_v_vec(IteratorFilters::LocallyOwnedCell(),
                 vertex_vec_handler_ref.begin_active());

    // start cell loop
    //        for (; cell!=endc; ++cell, ++cell_v_vec)
    for (; cell != endc; ++cell, ++cell_v, ++cell_v_vec)
      {
        Assert(cell->is_locally_owned(), ExcInternalError());
        Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

        const UpdateFlags uf_face(update_quadrature_points |
                                  update_normal_vectors | update_values |
                                  update_JxW_values | update_gradients);
        FEFaceValues<dim> fe_face_values_ref(mapping, fe, qf_face, uf_face);

        // Start loop over faces in element
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            if (cell->face(face)->at_boundary() == true)
              {
                fe_face_values_ref.reinit(cell, face);

                // Get displacement gradients for current face
                std::vector<Tensor<2, dim>> solution_grads_u_f(n_q_points_f);
                fe_face_values_ref[u_fe].get_function_gradients(
                  solution_total, solution_grads_u_f);

                // start gauss point loop
                for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                     ++f_q_point)
                  {
                    // Compute deformation gradient from displacements gradient
                    //(present configuration)
                    const Tensor<2, dim, ADNumberType> F_AD =
                      Physics::Elasticity::Kinematics::F(
                        solution_grads_u_f[f_q_point]);
                    ADNumberType det_F_AD = determinant(F_AD);
                    double       det_F    = Tensor<0, dim, double>(det_F_AD);

                    // Compute load vectors derived from Neumann bcs on surface
                    const Tensor<1, dim> &N =
                      fe_face_values_ref.normal_vector(f_q_point);
                    const Point<dim> &pt =
                      fe_face_values_ref.quadrature_point(f_q_point);
                    const Tensor<1, dim> traction =
                      get_neumann_traction(cell->face(face)->boundary_id(),
                                           pt,
                                           N);

                    //                      if (traction.norm()<1e-12) continue;

                    // OUTPUT AVERAGED ON ELEMENTS
                    // -------------------------------------------
                    if (parameters.outtype == "elements")
                      {
                        for (unsigned int j = 0; j < dim; ++j)
                          {
                            loads_elements[j](cell->active_cell_index()) +=
                              traction[j] / n_q_points_f;
                          }
                      }
                    // OUTPUT AVERAGED ON NODES
                    // -------------------------------------------
                    else if (parameters.outtype == "nodes")
                      {
                        for (unsigned int v = 0;
                             v < (GeometryInfo<dim>::vertices_per_face);
                             ++v)
                          {
                            types::global_dof_index local_vertex_indices =
                              cell_v->face(face)->vertex_dof_index(v, 0);
                            counter_on_vertices_mpi(local_vertex_indices) += 1;

                            for (unsigned int k = 0; k < dim; ++k)
                              {
                                types::global_dof_index
                                  local_vertex_vec_indices =
                                    cell_v_vec->face(face)->vertex_dof_index(v,
                                                                             k);
                                counter_on_vertices_vec_mpi(
                                  local_vertex_vec_indices) += 1;
                                loads_vertex_vec_mpi(
                                  local_vertex_vec_indices) += traction[k];
                              }
                            jacobian_vertex_mpi(local_vertex_indices) += det_F;
                          }
                      }
                    //--------------------------------------------------------------
                  } // end gauss point loop
              }     // end if face is in boundary
          }         // end face loop
      }             // end cell loop

    if (parameters.outtype == "nodes")
      {
        for (unsigned int d = 0; d < (vertex_handler_ref.n_dofs()); ++d)
          {
            sum_counter_on_vertices[d] =
              Utilities::MPI::sum(counter_on_vertices_mpi[d], mpi_communicator);
            //        		std::cout << "1: " << sum_counter_on_vertices[d] <<
            //        std::endl;
            sum_jacobian_vertex[d] =
              Utilities::MPI::sum(jacobian_vertex_mpi[d], mpi_communicator);
            //        		std::cout << "2: " << sum_jacobian_vertex[d] <<
            //        std::endl;
          }
        for (unsigned int d = 0; d < (vertex_vec_handler_ref.n_dofs()); ++d)
          {
            sum_counter_on_vertices_vec[d] =
              Utilities::MPI::sum(counter_on_vertices_vec_mpi[d],
                                  mpi_communicator);
            sum_loads_vertex_vec[d] =
              Utilities::MPI::sum(loads_vertex_vec_mpi[d], mpi_communicator);
          }

        for (unsigned int d = 0; d < (vertex_handler_ref.n_dofs()); ++d)
          {
            if (sum_counter_on_vertices[d] > 0)
              {
                sum_jacobian_vertex[d] /= sum_counter_on_vertices[d];
                //            		std::cout << "3: " << sum_jacobian_vertex[d]
                //            << std::endl;
              }
          }

        for (unsigned int d = 0; d < (vertex_vec_handler_ref.n_dofs()); ++d)
          {
            if (sum_counter_on_vertices_vec[d] > 0)
              {
                sum_loads_vertex_vec[d] /= sum_counter_on_vertices_vec[d];
              }
          }
      }

    DataOutFaces<dim> data_out_face;
    //        DataOutBase::VtkFlags flags;
    //         flags.write_higher_order_cells = true;
    //         data_out_face.set_flags(flags);
    // FilteredDataOutFaces<dim> data_out_face;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      face_comp_type(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    face_comp_type.push_back(DataComponentInterpretation::component_is_scalar);

    GridTools::get_subdomain_association(triangulation, partition_int);

    std::vector<std::string> ouput_name_face(dim, "displacement");
    ouput_name_face.push_back("pore_pressure");

    data_out_face.attach_dof_handler(dof_handler_ref);
    data_out_face.add_data_vector(solution_total,
                                  ouput_name_face,
                                  DataOutFaces<dim>::type_dof_data,
                                  face_comp_type);


    const Vector<double> partitioning(partition_int.begin(),
                                      partition_int.end());

    data_out_face.add_data_vector(partitioning, "partitioning");

    // Integration point results
    // -----------------------------------------------------------
    if (parameters.outtype == "elements")
      {
        data_out_face.add_data_vector(loads_elements[0], "load_x");
        data_out_face.add_data_vector(loads_elements[1], "load_y");
        data_out_face.add_data_vector(loads_elements[2], "load_z");
      }
    else if (parameters.outtype == "nodes")
      {
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          face_comp_type_vec(
            dim, DataComponentInterpretation::component_is_part_of_vector);
        std::vector<std::string> ouput_name_face_vec(dim, "load");

        data_out_face.add_data_vector(vertex_vec_handler_ref,
                                      sum_loads_vertex_vec,
                                      ouput_name_face_vec,
                                      face_comp_type_vec);
        data_out_face.add_data_vector(vertex_handler_ref,
                                      sum_jacobian_vertex,
                                      "jacobian");
      }
    //---------------------------------------------------------------------

    data_out_face.build_patches(degree_displ);
    // data_out_face.build_patches(mapping, degree_displ,
    // DataOutFaces<dim>::curved_boundary); data_out_face.build_patches();
    struct Filename_faces
    {
      static std::string
      get_filename_face_vtu(unsigned int       process,
                            unsigned int       timestep,
                            const unsigned int n_digits = 5)
      {
        std::ostringstream filename_face_vtu;
        filename_face_vtu << "bcs."
                          << Utilities::int_to_string(process, n_digits) << "."
                          << Utilities::int_to_string(timestep, n_digits)
                          << ".vtu";
        return filename_face_vtu.str();
      }

      static std::string
      get_filename_face_pvtu(unsigned int       timestep,
                             const unsigned int n_digits = 5)
      {
        std::ostringstream filename_face_vtu;
        filename_face_vtu << "bcs."
                          << Utilities::int_to_string(timestep, n_digits)
                          << ".pvtu";
        return filename_face_vtu.str();
      }

      static std::string
      get_filename_face_pvd(void)
      {
        std::ostringstream filename_face_vtu;
        filename_face_vtu << "bcs.pvd";
        return filename_face_vtu.str();
      }
    };

    const std::string filename_face_vtu =
      Filename_faces::get_filename_face_vtu(this_mpi_process, timestep);
    std::ofstream output_face(parameters.output_directory + "/" +
                              filename_face_vtu.c_str());
    data_out_face.write_vtu(output_face);

    // We have a collection of files written in parallel
    // This next set of steps should only be performed by master process
    if (this_mpi_process == 0)
      {
        // List of all files written out at this timestep by all processors
        std::vector<std::string> parallel_filenames_face_vtu;
        for (unsigned int p = 0; p < n_mpi_processes; ++p)
          {
            parallel_filenames_face_vtu.push_back(
              Filename_faces::get_filename_face_vtu(p, timestep));
          }

        const std::string filename_face_pvtu(
          Filename_faces::get_filename_face_pvtu(timestep));
        std::ofstream pvtu_master(parameters.output_directory + "/" +
                                  filename_face_pvtu.c_str());
        data_out_face.write_pvtu_record(pvtu_master,
                                        parallel_filenames_face_vtu);

        // Time dependent data master file
        static std::vector<std::pair<double, std::string>>
          time_and_name_history_face;
        time_and_name_history_face.push_back(
          std::make_pair(current_time, filename_face_pvtu));
        const std::string filename_face_pvd(
          Filename_faces::get_filename_face_pvd());
        std::ofstream pvd_output_face(parameters.output_directory + "/" +
                                      filename_face_pvd.c_str());
        DataOutBase::write_pvd_record(pvd_output_face,
                                      time_and_name_history_face);
      }
  }

  // Print results to plotting file
  template <int dim>
  void
  Solid<dim>::output_results_to_plot(
    const unsigned int                 timestep,
    const double                       current_time,
    TrilinosWrappers::MPI::BlockVector solution_IN,
    std::vector<Point<dim>>           &tracked_vertices_IN,
    std::ofstream                     &plotpointfile) const
  {
    TrilinosWrappers::MPI::BlockVector solution_total(
      locally_owned_partitioning,
      locally_relevant_partitioning,
      mpi_communicator,
      false);

    (void)timestep;
    solution_total = solution_IN;

    // Variables needed to print the solution file for plotting
    Point<dim>                  reaction_force;
    Point<dim>                  reaction_force_pressure;
    Point<dim>                  reaction_force_extra;
    Point<dim>                  reaction_force_extra_base;
    Point<dim>                  reaction_force_extra_ext_func;
    double                      total_fluid_flow          = 0.0;
    double                      total_porous_dissipation  = 0.0;
    double                      total_viscous_dissipation = 0.0;
    double                      total_solid_vol           = 0.0;
    double                      total_vol_current         = 0.0;
    double                      total_vol_reference       = 0.0;
    std::vector<Point<dim + 1>> solution_vertices(tracked_vertices_IN.size());
    double                      reaction_torque  = 0.0;
    double                      det_F_min_cells  = 1.0;
    double                      det_F_min_faces  = 1.0;
    double                      seepage_vec_mean = 0.0;
    (void)seepage_vec_mean;

    // Auxiliar variables needed for mpi processing
    Tensor<1, dim> sum_reaction_mpi;
    Tensor<1, dim> sum_reaction_pressure_mpi;
    Tensor<1, dim> sum_reaction_extra_mpi;
    Tensor<1, dim> sum_reaction_extra_base_mpi;
    Tensor<1, dim> sum_reaction_extra_ext_func_mpi;
    sum_reaction_mpi                                = 0.0;
    sum_reaction_pressure_mpi                       = 0.0;
    sum_reaction_extra_mpi                          = 0.0;
    sum_reaction_extra_base_mpi                     = 0.0;
    sum_reaction_extra_ext_func_mpi                 = 0.0;
    double              sum_total_flow_mpi          = 0.0;
    double              sum_porous_dissipation_mpi  = 0.0;
    double              sum_viscous_dissipation_mpi = 0.0;
    double              sum_solid_vol_mpi           = 0.0;
    double              sum_vol_current_mpi         = 0.0;
    double              sum_vol_reference_mpi       = 0.0;
    double              sum_torque_mpi              = 0.0;
    double              det_F_min_cells_mpi         = 1.0;
    double              det_F_min_faces_mpi         = 1.0;
    std::vector<double> det_F_cells_mpi;
    std::vector<double> det_F_faces_mpi{
      100}; // Had to put a unsreasonal high value (which does not matter since
            // we search for minima) to avoid segfault in std::min_element
    // std::vector<double> seepage_vec_mpi;
    double seepage_vec_mean_mpi = 0.0;
    (void)seepage_vec_mean_mpi;

    // Declare an instance of the material class object
    if (parameters.mat_type == "Neo-Hooke")
      NeoHooke<dim, ADNumberType> material(parameters, time);
    else if (parameters.mat_type == "Neo-Hooke-PS")
      NeoHookePS<dim, ADNumberType> material(parameters, time);
    else if (parameters.mat_type == "Neo-Hooke-Ehlers")
      NeoHookeEhlers<dim, ADNumberType> material(parameters, time);
    else if (parameters.mat_type == "Ogden")
      Ogden<dim, ADNumberType> material(parameters, time);
    else if (parameters.mat_type == "visco-Ogden")
      visco_Ogden<dim, ADNumberType> material(parameters, time);
    else if (parameters.mat_type == "visco2-Ogden")
      visco2_Ogden<dim, ADNumberType> material(parameters, time);
    else
      Assert(false, ExcMessage("Material type not implemented"));

    // Define a local instance of FEValues to compute updated values required
    // to calculate stresses
    const UpdateFlags uf_cell(update_values | update_gradients |
                              update_JxW_values | update_quadrature_points);
    FEValues<dim>     fe_values_ref(mapping, fe, qf_cell, uf_cell);

    // Iterate through elements (cells) and Gauss Points
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator> cell(
      IteratorFilters::LocallyOwnedCell(), dof_handler_ref.begin_active()),
      endc(IteratorFilters::LocallyOwnedCell(), dof_handler_ref.end());
    // start cell loop
    for (; cell != endc; ++cell)
      {
        Assert(cell->is_locally_owned(), ExcInternalError());
        Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

        fe_values_ref.reinit(cell);

        std::vector<Tensor<2, dim>> solution_grads_u(n_q_points);
        fe_values_ref[u_fe].get_function_gradients(solution_total,
                                                   solution_grads_u);

        std::vector<double> solution_values_p_fluid_total(n_q_points);
        fe_values_ref[p_fluid_fe].get_function_values(
          solution_total, solution_values_p_fluid_total);

        std::vector<Tensor<1, dim>> solution_grads_p_fluid_AD(n_q_points);
        fe_values_ref[p_fluid_fe].get_function_gradients(
          solution_total, solution_grads_p_fluid_AD);

        // start gauss point loop
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const Tensor<2, dim, ADNumberType> F_AD =
              Physics::Elasticity::Kinematics::F(solution_grads_u[q_point]);
            ADNumberType det_F_AD = determinant(F_AD);
            const double det_F    = Tensor<0, dim, double>(det_F_AD);
            det_F_cells_mpi.push_back(det_F);

            const std::vector<
              std::shared_ptr<const PointHistory<dim, ADNumberType>>>
              lqph = quadrature_point_history.get_data(cell);
            Assert(lqph.size() == n_q_points, ExcInternalError());

            double JxW = fe_values_ref.JxW(q_point);

            // Volumes
            sum_vol_current_mpi += det_F * JxW;
            sum_vol_reference_mpi += JxW;
            // sum_solid_vol_mpi += parameters.solid_vol_frac * JxW * det_F;
            sum_solid_vol_mpi += parameters.solid_vol_frac * JxW;

            // Seepage velocity
            const Tensor<2, dim, ADNumberType> F_inv = invert(F_AD);
            const Tensor<1, dim, ADNumberType> grad_p_fluid_AD =
              solution_grads_p_fluid_AD[q_point] * F_inv;
            const Tensor<1, dim, ADNumberType> seepage_vel_AD =
              lqph[q_point]->get_seepage_velocity_current(F_AD,
                                                          grad_p_fluid_AD);

            Tensor<1, dim> seepage;
            for (unsigned int i = 0; i < dim; ++i)
              seepage[i] = Tensor<0, dim, double>(seepage_vel_AD[i]);

            /*//test
            SymmetricTensor<2,dim> sigma_E_ext_func;
            const SymmetricTensor<2,dim,ADNumberType> sigma_E_ext_func_AD =
        lqph[q_point]->get_Cauchy_E_ext_func(F_AD);

            double det_F_converged = lqph[q_point]->get_converged_det_F();

            for (unsigned int i=0; i<dim; ++i)
              for (unsigned int j=0; j<dim; ++j) {
                sigma_E_ext_func[i][j] =
        Tensor<0,dim,double>(sigma_E_ext_func_AD[i][j]);
              }

            const Point<dim> gauss_coord2 =
        fe_values_ref.quadrature_point(q_point); std::ofstream
        sigma_ext_func_cells; sigma_ext_func_cells.open("sigma_ext_func_cells",
        std::ofstream::app); sigma_ext_func_cells << std::setprecision(8) <<
        std::scientific; sigma_ext_func_cells << std::setw(16) <<
        this->time->get_current() << ","
                << std::setw(16) << gauss_coord2[0] << ","
        << std::setw(16) << gauss_coord2[1] << ","
        << std::setw(16) << gauss_coord2[2] << ","
        << std::setw(16) << JxW << ","
        << std::setw(16) << det_F << ","
        << std::setw(16) << det_F_converged << ","
        << std::setw(16) << sigma_E_ext_func[0][0] << ","
        << std::setw(16) << sigma_E_ext_func[1][1] << ","
        << std::setw(16) << sigma_E_ext_func[2][2] << std::endl;
            sigma_ext_func_cells.close();

            //if (seepage[2]>0)
            //	std::cout << seepage[2] << " cell loop" << std::endl;

            //const Point<dim> gauss_coord =
        fe_values_ref.quadrature_point(q_point);
            //if (gauss_coord[2] < 0.1) {
            //	seepage_vec_mpi.push_back (seepage[2]);
              //std::cout << seepage[2] << std::endl;
            //}*/

            // Dissipations
            const double porous_dissipation =
              lqph[q_point]->get_porous_dissipation(F_AD, grad_p_fluid_AD);
            sum_porous_dissipation_mpi += porous_dissipation * det_F * JxW;

            const double viscous_dissipation =
              lqph[q_point]->get_viscous_dissipation();
            sum_viscous_dissipation_mpi += viscous_dissipation * det_F * JxW;

            //---------------------------------------------------------------
          } // end gauss point loop

        // Compute reaction force on load boundary & total fluid flow across
        // drained boundary.
        // Define a local instance of FEFaceValues to compute values required
        // to calculate reaction force
        const UpdateFlags uf_face(update_values | update_gradients |
                                  update_normal_vectors | update_JxW_values |
                                  update_quadrature_points);
        FEFaceValues<dim> fe_face_values_ref(mapping, fe, qf_face, uf_face);

        // start face loop
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            // Reaction force
            if (cell->face(face)->at_boundary() == true &&
                cell->face(face)->boundary_id() ==
                  get_reaction_boundary_id_for_output())
              {
                fe_face_values_ref.reinit(cell, face);

                // Get displacement gradients for current face
                std::vector<Tensor<2, dim>> solution_grads_u_f(n_q_points_f);
                fe_face_values_ref[u_fe].get_function_gradients(
                  solution_total, solution_grads_u_f);

                // Get pressure for current element
                std::vector<double> solution_values_p_fluid_total_f(
                  n_q_points_f);
                fe_face_values_ref[p_fluid_fe].get_function_values(
                  solution_total, solution_values_p_fluid_total_f);

                // start gauss points on faces loop
                for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                     ++f_q_point)
                  {
                    const Tensor<1, dim> &N =
                      fe_face_values_ref.normal_vector(f_q_point);
                    const double JxW_f = fe_face_values_ref.JxW(f_q_point);

                    // Compute deformation gradient from displacements gradient
                    //(present configuration)
                    const Tensor<2, dim, ADNumberType> F_AD =
                      Physics::Elasticity::Kinematics::F(
                        solution_grads_u_f[f_q_point]);
                    ADNumberType det_F_AD = determinant(F_AD);
                    // double       det_F    = Tensor<0, dim, double>(det_F_AD);

                    const std::vector<
                      std::shared_ptr<const PointHistory<dim, ADNumberType>>>
                      lqph = quadrature_point_history.get_data(cell);
                    Assert(lqph.size() == n_q_points, ExcInternalError());

                    const double p_fluid =
                      solution_values_p_fluid_total[f_q_point];

                    // Cauchy stress
                    static const SymmetricTensor<2, dim, double> I(
                      Physics::Elasticity::StandardTensors<dim>::I);
                    SymmetricTensor<2, dim> sigma_E;
                    SymmetricTensor<2, dim> sigma_E_base;
                    SymmetricTensor<2, dim> sigma_E_ext_func;
                    const SymmetricTensor<2, dim, ADNumberType> sigma_E_AD =
                      lqph[f_q_point]->get_Cauchy_E(F_AD);
                    const SymmetricTensor<2, dim, ADNumberType>
                      sigma_E_base_AD =
                        lqph[f_q_point]->get_Cauchy_E_base(F_AD);
                    const SymmetricTensor<2, dim, ADNumberType>
                      sigma_E_ext_func_AD =
                        lqph[f_q_point]->get_Cauchy_E_ext_func(F_AD);

                    // double det_F_converged =
                    //   lqph[f_q_point]->get_converged_det_F();

                    for (unsigned int i = 0; i < dim; ++i)
                      for (unsigned int j = 0; j < dim; ++j)
                        {
                          sigma_E[i][j] =
                            Tensor<0, dim, double>(sigma_E_AD[i][j]);
                          sigma_E_base[i][j] =
                            Tensor<0, dim, double>(sigma_E_base_AD[i][j]);
                          sigma_E_ext_func[i][j] =
                            Tensor<0, dim, double>(sigma_E_ext_func_AD[i][j]);
                        }

                    SymmetricTensor<2, dim> sigma_fluid_vol(I);
                    sigma_fluid_vol *= -1.0 * p_fluid;
                    const SymmetricTensor<2, dim> sigma =
                      sigma_E + sigma_fluid_vol;
                    sum_reaction_mpi += sigma * N * JxW_f;
                    sum_reaction_pressure_mpi += sigma_fluid_vol * N * JxW_f;
                    sum_reaction_extra_mpi += sigma_E * N * JxW_f;
                    sum_reaction_extra_base_mpi += sigma_E_base * N * JxW_f;
                    sum_reaction_extra_ext_func_mpi +=
                      sigma_E_ext_func * N * JxW_f;


                    /*const Point<dim> gauss_coord2 =
            fe_face_values_ref.quadrature_point(f_q_point); std::ofstream
            sigma_ext_func_reaction_force_boundary;
                    sigma_ext_func_reaction_force_boundary.open("sigma_ext_func_reaction_force_boundary",
            std::ofstream::app); sigma_ext_func_reaction_force_boundary <<
            std::setprecision(8) << std::scientific;
                    sigma_ext_func_reaction_force_boundary << std::setw(16) <<
            this->time->get_current() << ","
                        << std::setw(16) << gauss_coord2[0] << ","
            << std::setw(16) << gauss_coord2[1] << ","
            << std::setw(16) << gauss_coord2[2] << ","
            << std::setw(16) << JxW_f << ","
            << std::setw(16) << det_F << ","
            << std::setw(16) << det_F_converged << ","
                        << std::setw(16) << sigma_E_ext_func[0][0] << ","
                        << std::setw(16) << sigma_E_ext_func[1][1] << ","
            << std::setw(16) << sigma_E_ext_func[2][2] << std::endl;
                    sigma_ext_func_reaction_force_boundary.close();*/

                    // Transform components of Cauchy stresses into cylindrical
                    // coordinates for torque evaluation under torsional shear
                    // loading

                    // Obtain spherical coordinates from current Gauss Point
                    const std::array<double, dim> gauss_sph =
                      GeometricUtilities::Coordinates::to_spherical(
                        fe_face_values_ref.quadrature_point(f_q_point));
                    const Point<dim> gauss_coord =
                      fe_face_values_ref.quadrature_point(f_q_point);
                    // Compute azimuth angle and radius
                    const double theta = gauss_sph[1];
                    // std::cout << theta << std::endl;
                    const double r =
                      sqrt(pow(gauss_coord[0], 2) + pow(gauss_coord[1], 2));
                    // Define a new 2nd order tensor to store the cylindrical
                    // coordinates of sigma_E
                    SymmetricTensor<2, dim> sigma_E_cyl;

                    // Compute the components of sigma_E_cyl associated with the
                    // corresponding cylindrical coordinate system from sigma_E
                    sigma_E_cyl[0][0] =
                      sigma_E[0][0] * pow(cos(theta), 2) +
                      sigma_E[1][1] * pow(sin(theta), 2) +
                      2 * sigma_E[0][1] * sin(theta) * cos(theta); // sigma_r_r
                    sigma_E_cyl[1][1] = sigma_E[0][0] * pow(sin(theta), 2) +
                                        sigma_E[1][1] * pow(cos(theta), 2) -
                                        2 * sigma_E[0][1] * sin(theta) *
                                          cos(theta); // sigma_theta_theta
                    sigma_E_cyl[0][1] =
                      (sigma_E[1][1] - sigma_E[0][0]) * sin(theta) *
                        cos(theta) +
                      sigma_E[0][1] * (pow(cos(theta), 2) -
                                       pow(sin(theta), 2)); // sigma_r_theta
                    sigma_E_cyl[0][2] = sigma_E[0][2] * cos(theta) +
                                        sigma_E[1][2] * sin(theta); // sigma_r_z
                    sigma_E_cyl[1][2] =
                      sigma_E[1][2] * cos(theta) -
                      sigma_E[0][2] * sin(theta);          // sigma_theta_z
                    sigma_E_cyl[2][2] = sigma_E[2][2];     // sigma_zz
                    sigma_E_cyl[1][0] = sigma_E_cyl[0][1]; // sigma_theta_r
                    sigma_E_cyl[2][0] = sigma_E_cyl[0][2]; // sigma_z_r
                    sigma_E_cyl[2][1] = sigma_E_cyl[1][2]; // sigma_z_theta

                    // Analogeously define and compute cylindrical components of
                    // sigma_fluid_vol
                    SymmetricTensor<2, dim> sigma_fluid_vol_cyl;
                    sigma_fluid_vol_cyl[0][0] =
                      sigma_fluid_vol[0][0] * pow(cos(theta), 2) +
                      sigma_fluid_vol[1][1] * pow(sin(theta), 2) +
                      2 * sigma_fluid_vol[0][1] * sin(theta) *
                        cos(theta); // sigma_r_r
                    sigma_fluid_vol_cyl[1][1] =
                      sigma_fluid_vol[0][0] * pow(sin(theta), 2) +
                      sigma_fluid_vol[1][1] * pow(cos(theta), 2) -
                      2 * sigma_fluid_vol[0][1] * sin(theta) *
                        cos(theta); // sigma_theta_theta
                    sigma_fluid_vol_cyl[0][1] =
                      (sigma_fluid_vol[1][1] - sigma_fluid_vol[0][0]) *
                        sin(theta) * cos(theta) +
                      sigma_fluid_vol[0][1] *
                        (pow(cos(theta), 2) -
                         pow(sin(theta), 2)); // sigma_r_theta
                    sigma_fluid_vol_cyl[0][2] =
                      sigma_fluid_vol[0][2] * cos(theta) +
                      sigma_fluid_vol[1][2] * sin(theta); // sigma_r_z
                    sigma_fluid_vol_cyl[1][2] =
                      sigma_fluid_vol[1][2] * cos(theta) -
                      sigma_fluid_vol[0][2] * sin(theta); // sigma_theta_z
                    sigma_fluid_vol_cyl[2][2] =
                      sigma_fluid_vol[2][2]; // sigma_zz
                    sigma_fluid_vol_cyl[1][0] =
                      sigma_fluid_vol_cyl[0][1]; // sigma_theta_r
                    sigma_fluid_vol_cyl[2][0] =
                      sigma_fluid_vol_cyl[0][2]; // sigma_z_r
                    sigma_fluid_vol_cyl[2][1] =
                      sigma_fluid_vol_cyl[1][2]; // sigma_z_theta

                    // Add solid and fluid component to get total Cauchy stress
                    const SymmetricTensor<2, dim> sigma_cyl =
                      sigma_E_cyl + sigma_fluid_vol_cyl;

                    // Compute torque contribution from z_theta component, the
                    // according weighted area JxW_f and the distance r to
                    // center of rotation
                    sum_torque_mpi += sigma_cyl[2][1] * r * JxW_f;
                  } // end gauss points on faces loop
              }



            // Fluid flow
            if (cell->face(face)->at_boundary() == true &&
                (cell->face(face)->boundary_id() ==
                   get_drained_boundary_id_for_output().first ||
                 cell->face(face)->boundary_id() ==
                   get_drained_boundary_id_for_output().second))
              {
                fe_face_values_ref.reinit(cell, face);

                // Get displacement gradients for current face
                std::vector<Tensor<2, dim>> solution_grads_u_f(n_q_points_f);
                fe_face_values_ref[u_fe].get_function_gradients(
                  solution_total, solution_grads_u_f);

                // Get pressure gradients for current face
                std::vector<Tensor<1, dim>> solution_grads_p_f(n_q_points_f);
                fe_face_values_ref[p_fluid_fe].get_function_gradients(
                  solution_total, solution_grads_p_f);

                // start gauss points on faces loop
                for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                     ++f_q_point)
                  {
                    const Tensor<1, dim> &N =
                      fe_face_values_ref.normal_vector(f_q_point);
                    const double JxW_f = fe_face_values_ref.JxW(f_q_point);

                    // Deformation gradient and inverse from displacements
                    // gradient (present configuration)
                    const Tensor<2, dim, ADNumberType> F_AD =
                      Physics::Elasticity::Kinematics::F(
                        solution_grads_u_f[f_q_point]);

                    const Tensor<2, dim, ADNumberType> F_inv_AD = invert(F_AD);
                    ADNumberType det_F_AD = determinant(F_AD);
                    const Tensor<2, dim, ADNumberType> F_inv_AD_trans =
                      transpose(F_inv_AD); // added-1.10.21

                    const std::vector<
                      std::shared_ptr<const PointHistory<dim, ADNumberType>>>
                      lqph = quadrature_point_history.get_data(cell);
                    Assert(lqph.size() == n_q_points, ExcInternalError());

                    // Seepage velocity
                    Tensor<1, dim> seepage;
                    // double det_F = Tensor<0,dim,double>(det_F_AD);
                    const Tensor<1, dim, ADNumberType> grad_p =
                      solution_grads_p_f[f_q_point] * F_inv_AD;
                    const Tensor<1, dim, ADNumberType> seepage_AD =
                      lqph[f_q_point]->get_seepage_velocity_current(F_AD,
                                                                    grad_p);

                    for (unsigned int i = 0; i < dim; ++i)
                      seepage[i] = Tensor<0, dim, double>(seepage_AD[i]);

                    // sum_total_flow_mpi += seepage * N * JxW_f;
                    // if (seepage[2]>0)
                    //	std::cout << seepage[2] << " face loop" << std::endl;
                    // sum_total_flow_mpi += (seepage/det_F) * N * JxW_f;
                    const Tensor<1, dim> temp1 =
                      contract<0, 0>(seepage, F_inv_AD_trans);
                    sum_total_flow_mpi += temp1 * N * JxW_f; // added-1.10.21
                  } // end gauss points on faces loop
              }

            // Minimal Jacobian
            if (cell->face(face)->at_boundary() == true &&
                cell->face(face)->boundary_id() ==
                  get_reaction_boundary_id_for_output())
              {
                fe_face_values_ref.reinit(cell, face);

                // Get displacement gradients for current face
                std::vector<Tensor<2, dim>> solution_grads_u_f(n_q_points_f);
                fe_face_values_ref[u_fe].get_function_gradients(
                  solution_total, solution_grads_u_f);

                // start Gauss points loop on faces
                for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                     ++f_q_point)
                  {
                    // Compute deformation gradient from displacements gradient
                    // and its Jacobian
                    const Tensor<2, dim, ADNumberType> F_AD =
                      Physics::Elasticity::Kinematics::F(
                        solution_grads_u_f[f_q_point]);
                    ADNumberType det_F_AD = determinant(F_AD);
                    double       det_F    = Tensor<0, dim, double>(det_F_AD);
                    det_F_faces_mpi.push_back(det_F);
                  }
              }
          } // end face loop
      }     // end cell loop

    // Sum the results from different MPI process and then add to the
    // reaction_force vector In theory, the solution on each surface (each cell)
    // only exists in one MPI process so, we add all MPI process, one will have
    // the solution and the others will be zero
    for (unsigned int d = 0; d < dim; ++d)
      {
        reaction_force[d] =
          Utilities::MPI::sum(sum_reaction_mpi[d], mpi_communicator);
        reaction_force_pressure[d] =
          Utilities::MPI::sum(sum_reaction_pressure_mpi[d], mpi_communicator);
        reaction_force_extra[d] =
          Utilities::MPI::sum(sum_reaction_extra_mpi[d], mpi_communicator);
        reaction_force_extra_base[d] =
          Utilities::MPI::sum(sum_reaction_extra_base_mpi[d], mpi_communicator);
        reaction_force_extra_ext_func[d] =
          Utilities::MPI::sum(sum_reaction_extra_ext_func_mpi[d],
                              mpi_communicator);
      }

    // Same for total fluid flow, and for porous and viscous dissipations
    total_fluid_flow =
      Utilities::MPI::sum(sum_total_flow_mpi, mpi_communicator);
    total_porous_dissipation =
      Utilities::MPI::sum(sum_porous_dissipation_mpi, mpi_communicator);
    total_viscous_dissipation =
      Utilities::MPI::sum(sum_viscous_dissipation_mpi, mpi_communicator);
    total_solid_vol = Utilities::MPI::sum(sum_solid_vol_mpi, mpi_communicator);
    total_vol_current =
      Utilities::MPI::sum(sum_vol_current_mpi, mpi_communicator);
    total_vol_reference =
      Utilities::MPI::sum(sum_vol_reference_mpi, mpi_communicator);
    reaction_torque = Utilities::MPI::sum(sum_torque_mpi, mpi_communicator);

    det_F_min_cells_mpi =
      *std::min_element(det_F_cells_mpi.begin(), det_F_cells_mpi.end());
    det_F_min_cells =
      Utilities::MPI::min(det_F_min_cells_mpi, mpi_communicator);
    det_F_min_faces_mpi =
      *std::min_element(det_F_faces_mpi.begin(), det_F_faces_mpi.end());
    det_F_min_faces =
      Utilities::MPI::min(det_F_min_faces_mpi, mpi_communicator);

    // if (seepage_vec_mpi.size()>1) {
    //	seepage_vec_mean_mpi = std::accumulate(seepage_vec_mpi.begin(),
    // seepage_vec_mpi.end(), 0.0) / seepage_vec_mpi.size(); 	std::cout <<
    // seepage_vec_mean_mpi << std::endl; 	seepage_vec_mean =
    // seepage_vec_mean_mpi;
    // }
    // seepage_vec_mean = Utilities::MPI::sum(seepage_vec_mean_mpi,
    // mpi_communicator) / 6;

    //  Extract solution for tracked vectors
    // Copying an MPI::BlockVector into MPI::Vector is not possible,
    // so we copy each block of MPI::BlockVector into an MPI::Vector
    // And then we copy the MPI::Vector into "normal" Vectors
    TrilinosWrappers::MPI::Vector solution_vector_u_MPI(
      solution_total.block(u_block));
    TrilinosWrappers::MPI::Vector solution_vector_p_MPI(
      solution_total.block(p_fluid_block));
    Vector<double> solution_u_vector(solution_vector_u_MPI);
    Vector<double> solution_p_vector(solution_vector_p_MPI);

    if (this_mpi_process == 0)
      {
        // Append the pressure solution vector to the displacement solution
        // vector, creating a single solution vector equivalent to the original
        // BlockVector so FEFieldFunction will work with the dof_handler_ref.
        Vector<double> solution_vector(solution_p_vector.size() +
                                       solution_u_vector.size());

        for (unsigned int d = 0; d < (solution_u_vector.size()); ++d)
          solution_vector[d] = solution_u_vector[d];

        for (unsigned int d = 0; d < (solution_p_vector.size()); ++d)
          solution_vector[solution_u_vector.size() + d] = solution_p_vector[d];

        Functions::FEFieldFunction<dim, Vector<double>> find_solution(
          dof_handler_ref, solution_vector);

        for (unsigned int p = 0; p < tracked_vertices_IN.size(); ++p)
          {
            Vector<double> update(dim + 1);
            Point<dim>     pt_ref;

            pt_ref[0] = tracked_vertices_IN[p][0];
            pt_ref[1] = tracked_vertices_IN[p][1];
            pt_ref[2] = tracked_vertices_IN[p][2];

            find_solution.vector_value(pt_ref, update);

            for (unsigned int d = 0; d < (dim + 1); ++d)
              {
                // For values close to zero, set to 0.0
                if (abs(update[d]) < 1.5 * parameters.tol_u)
                  update[d] = 0.0;
                solution_vertices[p][d] = update[d];
              }
          }
        // Write the results to the plotting file.
        // Add two blank lines between cycles in the cyclic loading examples so
        // GNUPLOT can detect each cycle as a different block
        if ((parameters.geom_type ==
             "Budday_cube_tension_compression_fully_fixed") ||
            (parameters.geom_type == "Budday_cube_tension_compression") ||
            (parameters.geom_type == "Budday_cube_shear_fully_fixed"))
          {
            const double delta_time = time->get_delta_t();
            const double end_time   = time->get_end();

            // This was previously called from parameters.
            // Current time is passed into the function, maybe it can be called
            // from time-> too?

            if (((parameters.geom_type ==
                  "Budday_cube_tension_compression_fully_fixed") ||
                 (parameters.geom_type == "Budday_cube_tension_compression") ||
                 (parameters.geom_type == "Budday_cube_shear_fully_fixed")) &&
                ((abs(current_time - end_time / 3.) < 0.9 * delta_time) ||
                 (abs(current_time - 2. * end_time / 3.) < 0.9 * delta_time)) &&
                parameters.num_cycle_sets == 1)
              {
                plotpointfile << std::endl << std::endl;
              }
            if (((parameters.geom_type ==
                  "Budday_cube_tension_compression_fully_fixed") ||
                 (parameters.geom_type == "Budday_cube_tension_compression") ||
                 (parameters.geom_type == "Budday_cube_shear_fully_fixed")) &&
                ((abs(current_time - end_time / 9.) < 0.9 * delta_time) ||
                 (abs(current_time - 2. * end_time / 9.) < 0.9 * delta_time) ||
                 (abs(current_time - 3. * end_time / 9.) < 0.9 * delta_time) ||
                 (abs(current_time - 5. * end_time / 9.) < 0.9 * delta_time) ||
                 (abs(current_time - 7. * end_time / 9.) < 0.9 * delta_time)) &&
                parameters.num_cycle_sets == 2)
              {
                plotpointfile << std::endl << std::endl;
              }
          }

        plotpointfile << std::setprecision(6) << std::scientific;
        plotpointfile << std::setw(16) << current_time << "," << std::setw(15)
                      << total_vol_reference << "," << std::setw(15)
                      << total_vol_current << "," << std::setw(15)
                      << total_solid_vol << ",";

        if (current_time == 0.0)
          {
            for (unsigned int p = 0; p < tracked_vertices_IN.size(); ++p)
              {
                for (unsigned int d = 0; d < dim; ++d)
                  plotpointfile << std::setw(15) << 0.0 << ",";

                plotpointfile << std::setw(15) << parameters.drained_pressure
                              << ",";
              }
            for (unsigned int d = 0; d < (3 * dim + 2); ++d)
              plotpointfile << std::setw(15) << 0.0 << ",";

            plotpointfile << std::setw(15) << 0.0 << "," << std::setw(15) << 0.0
                          << ",";

            for (unsigned int d = 0; d < (2 * dim); ++d)
              plotpointfile << std::setw(15) << 0.0 << ",";

            plotpointfile << std::setw(15) << 1.0 << ",";
            plotpointfile << std::setw(15) << 1.0 << ",";
          }
        else
          {
            for (unsigned int p = 0; p < tracked_vertices_IN.size(); ++p)
              for (unsigned int d = 0; d < (dim + 1); ++d)
                plotpointfile << std::setw(15) << solution_vertices[p][d]
                              << ",";

            if (parameters.geom_type ==
                  "brain_rheometer_cyclic_tension_compression_exp_quarter" ||
                parameters.geom_type == "hydro_nano_graz_compression_exp_relax")
              {
                for (unsigned int d = 0; d < dim; ++d)
                  plotpointfile << std::setw(15) << reaction_force[d] * (4e-6)
                                << ",";
              }
            else
              {
                for (unsigned int d = 0; d < dim; ++d)
                  plotpointfile << std::setw(15) << reaction_force[d] << ",";
              }

            for (unsigned int d = 0; d < dim; ++d)
              plotpointfile << std::setw(15) << reaction_force_pressure[d]
                            << ",";

            for (unsigned int d = 0; d < dim; ++d)
              plotpointfile << std::setw(15) << reaction_force_extra[d] << ",";

            plotpointfile << std::setw(15) << total_fluid_flow
                          << ","
                          //<< std::setw(15) << seepage_vec_mean << ","
                          << std::setw(15) << total_porous_dissipation << ","
                          << std::setw(15) << total_viscous_dissipation << ","
                          << std::setw(15) << reaction_torque << ",";

            for (unsigned int d = 0; d < dim; ++d)
              plotpointfile << std::setw(15) << reaction_force_extra_base[d]
                            << ",";
            for (unsigned int d = 0; d < dim; ++d)
              plotpointfile << std::setw(15) << reaction_force_extra_ext_func[d]
                            << ",";

            plotpointfile << std::setw(15) << det_F_min_cells << ",";
            plotpointfile << std::setw(15) << det_F_min_faces << ",";
          }
        plotpointfile << std::endl;
      }
  }

  //    //Output results averaged on nodes to plotting file
  //    template <int dim>
  //    void Solid<dim>::output_results_averaged_on_nodes(
  //    	const unsigned int timestep,
  //		const double current_time,
  //		TrilinosWrappers::MPI::BlockVector solution_IN,
  //		std::vector<Point<dim> > &tracked_vertices_IN,
  //		std::ofstream &plotnodefile) const
  //		{
  //    		TrilinosWrappers::MPI::BlockVector
  //    solution_total(locally_owned_partitioning,
  //    														  locally_relevant_partitioning,
  //															  mpi_communicator,
  //															  false);
  //
  //    		(void) timestep;
  //    		solution_total = solution_IN;
  //
  //    		GradientPostprocessor<dim>
  //    gradient_postprocessor(p_fluid_component);
  //
  //    		//Declare local variables with number of stress components
  //    		//& assign value according to "dim" value
  //    		unsigned int num_comp_symm_tensor = 6;
  //
  //    		//Variables needed to print the solution file for plotting
  //    		Point<dim> reaction_force;
  //    		Point<dim> reaction_force_pressure;
  //    		Point<dim> reaction_force_extra;
  //    		Point<dim> reaction_force_extra_base;
  //    		Point<dim> reaction_force_extra_ext_func;
  //    		double total_fluid_flow = 0.0;
  //    		double total_porous_dissipation = 0.0;
  //    		double total_viscous_dissipation = 0.0;
  //    		double total_solid_vol = 0.0;
  //    		double total_vol_current = 0.0;
  //    		double total_vol_reference = 0.0;
  //    		std::vector<Point<dim+1>>
  //    solution_vertices(tracked_vertices_IN.size()); 		double reaction_torque
  //    = 0.0; 		double det_F_min = 1.0; 		double seepage_vec_mean = 0.0;
  //
  //    		//Auxiliar variables needed for mpi processing
  //    		Tensor<1,dim> sum_reaction_mpi;
  //    		Tensor<1,dim> sum_reaction_pressure_mpi;
  //    		Tensor<1,dim> sum_reaction_extra_mpi;
  //    		Tensor<1,dim> sum_reaction_extra_base_mpi;
  //    		Tensor<1,dim> sum_reaction_extra_ext_func_mpi;
  //    		sum_reaction_mpi = 0.0;
  //    		sum_reaction_pressure_mpi = 0.0;
  //    		sum_reaction_extra_mpi = 0.0;
  //    		sum_reaction_extra_base_mpi = 0.0;
  //    		sum_reaction_extra_ext_func_mpi = 0.0;
  //    		double sum_total_flow_mpi = 0.0;
  //    		double sum_porous_dissipation_mpi = 0.0;
  //    		double sum_viscous_dissipation_mpi = 0.0;
  //    		double sum_solid_vol_mpi = 0.0;
  //    		double sum_vol_current_mpi = 0.0;
  //    		double sum_vol_reference_mpi = 0.0;
  //    		double sum_torque_mpi = 0.0;
  //    		double det_F_min_mpi = 1.0;
  //    		std::vector<double> det_F_mpi;
  //    		std::vector<double> seepage_vec_mpi;
  //    		double seepage_vec_mean_mpi = 0.0;
  //
  //    		// OUTPUT AVERAGED ON NODES
  //    ----------------------------------------------
  //    		// We need to create a new FE space with a single dof per node to
  //    avoid
  //    		// duplication of the output on nodes for our problem with dim+1
  //    dofs. 		FE_Q<dim> fe_vertex(1); 		DoFHandler<dim>
  //    vertex_handler_ref(triangulation);
  //    		vertex_handler_ref.distribute_dofs(fe_vertex);
  //    		AssertThrow(vertex_handler_ref.n_dofs() ==
  //    triangulation.n_vertices(),
  //    					ExcDimensionMismatch(vertex_handler_ref.n_dofs(),
  //    					triangulation.n_vertices()));
  //
  //    		Vector<double> counter_on_vertices_mpi(vertex_handler_ref.n_dofs());
  //    		Vector<double> sum_counter_on_vertices(vertex_handler_ref.n_dofs());
  //
  //    		std::vector<Vector<double>> cauchy_stresses_total_vertex_mpi
  //										(num_comp_symm_tensor,
  //										 Vector<double>(vertex_handler_ref.n_dofs()));
  //    		std::vector<Vector<double>> sum_cauchy_stresses_total_vertex
  //										(num_comp_symm_tensor,
  //										 Vector<double>(vertex_handler_ref.n_dofs()));
  //    		std::vector<Vector<double>> cauchy_stresses_E_vertex_mpi
  //										(num_comp_symm_tensor,
  //										 Vector<double>(vertex_handler_ref.n_dofs()));
  //    		std::vector<Vector<double>> sum_cauchy_stresses_E_vertex
  //										(num_comp_symm_tensor,
  //										 Vector<double>(vertex_handler_ref.n_dofs()));
  //    		std::vector<Vector<double>> cauchy_stresses_E_ext_func_vertex_mpi
  //										(num_comp_symm_tensor,
  //										 Vector<double>(vertex_handler_ref.n_dofs()));
  //    		std::vector<Vector<double>> sum_cauchy_stresses_E_ext_func_vertex
  //										(num_comp_symm_tensor,
  //										 Vector<double>(vertex_handler_ref.n_dofs()));
  //
  //    		Vector<double>
  //    porous_dissipation_vertex_mpi(vertex_handler_ref.n_dofs());
  //    		Vector<double>
  //    sum_porous_dissipation_vertex(vertex_handler_ref.n_dofs());
  //    		Vector<double>
  //    viscous_dissipation_vertex_mpi(vertex_handler_ref.n_dofs());
  //    		Vector<double>
  //    sum_viscous_dissipation_vertex(vertex_handler_ref.n_dofs());
  //    		Vector<double>
  //    solid_vol_fraction_vertex_mpi(vertex_handler_ref.n_dofs());
  //    		Vector<double>
  //    sum_solid_vol_fraction_vertex(vertex_handler_ref.n_dofs());
  //    		Vector<double> jacobian_vertex_mpi(vertex_handler_ref.n_dofs());
  //    		Vector<double> sum_jacobian_vertex(vertex_handler_ref.n_dofs());
  //
  //    		// We need to create a new FE space with a dim dof per node to
  //			// be able to ouput data on nodes in vector form
  //			FESystem<dim> fe_vertex_vec(FE_Q<dim>(1),dim);
  //    		DoFHandler<dim> vertex_vec_handler_ref(triangulation);
  //    		vertex_vec_handler_ref.distribute_dofs(fe_vertex_vec);
  //    		AssertThrow(vertex_vec_handler_ref.n_dofs() ==
  //    (dim*triangulation.n_vertices()),
  //    					ExcDimensionMismatch(vertex_vec_handler_ref.n_dofs(),
  //    					(dim*triangulation.n_vertices())));
  //
  //    		Vector<double>
  //    seepage_velocity_vertex_vec_mpi(vertex_vec_handler_ref.n_dofs());
  //    		Vector<double>
  //    sum_seepage_velocity_vertex_vec(vertex_vec_handler_ref.n_dofs());
  //    		Vector<double>
  //    counter_on_vertices_vec_mpi(vertex_vec_handler_ref.n_dofs());
  //    		Vector<double>
  //    sum_counter_on_vertices_vec(vertex_vec_handler_ref.n_dofs());
  //    		//
  //    -----------------------------------------------------------------------
  //
  //    		//Declare and initialize local unit vectors (to construct tensor
  //    basis) 		std::vector<Tensor<1,dim>> basis_vectors (dim, Tensor<1,dim>()
  //    ); 		for (unsigned int i=0; i<dim; ++i) 			basis_vectors[i][i] = 1;
  //
  //    		//Declare an instance of the material class object
  //    		if (parameters.mat_type == "Neo-Hooke")
  //    			NeoHooke<dim,ADNumberType> material(parameters,time);
  //    		else if (parameters.mat_type == "Neo-Hooke-PS")
  //    			NeoHookePS<dim,ADNumberType> material(parameters,time);
  //    		else if (parameters.mat_type == "Neo-Hooke-Ehlers")
  //    			NeoHookeEhlers<dim,ADNumberType> material(parameters,time);
  //    		else if (parameters.mat_type == "Ogden")
  //    			Ogden<dim,ADNumberType> material(parameters, time);
  //    		else if (parameters.mat_type == "visco-Ogden")
  //    			visco_Ogden <dim,ADNumberType>material(parameters,time);
  //    		else if (parameters.mat_type == "visco2-Ogden")
  //    			visco2_Ogden <dim,ADNumberType>material(parameters,time);
  //    		else
  //    			Assert (false, ExcMessage("Material type not implemented"));
  //
  //    		//Define a local instance of FEValues to compute updated values
  //    required
  //    		//to calculate stresses
  //    		const UpdateFlags uf_cell(update_values | update_gradients |
  //    								  update_JxW_values | update_quadrature_points);
  //    		FEValues<dim> fe_values_ref (mapping, fe, qf_cell, uf_cell);
  //
  //    		//Iterate through elements (cells) and Gauss Points
  //    		FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
  //    		cell(IteratorFilters::LocallyOwnedCell(),
  //    			 dof_handler_ref.begin_active()),
  //			endc(IteratorFilters::LocallyOwnedCell(),
  //				 dof_handler_ref.end());
  //    		cell_v(IteratorFilters::LocallyOwnedCell(),
  //    		       vertex_handler_ref.begin_active()),
  //    		cell_v_vec(IteratorFilters::LocallyOwnedCell(),
  //    		           vertex_vec_handler_ref.begin_active());
  //		  //start cell loop
  //		  for (; cell!=endc; ++cell)
  //		  {
  //			  Assert(cell->is_locally_owned(), ExcInternalError());
  //			  Assert(cell->subdomain_id() == this_mpi_process,
  // ExcInternalError());
  //
  //			  fe_values_ref.reinit(cell);
  //
  //			  std::vector<Tensor<2,dim>> solution_grads_u(n_q_points);
  //			  fe_values_ref[u_fe].get_function_gradients(solution_total,
  //					  solution_grads_u);
  //
  //			  std::vector<double> solution_values_p_fluid_total(n_q_points);
  //			  fe_values_ref[p_fluid_fe].get_function_values(solution_total,
  //					  solution_values_p_fluid_total);
  //
  //			  std::vector<Tensor<1,dim >> solution_grads_p_fluid_AD(n_q_points);
  //			  fe_values_ref[p_fluid_fe].get_function_gradients(solution_total,
  //					  solution_grads_p_fluid_AD);
  //
  //			  //start gauss point loop
  //			  for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
  //			  {
  //				  const Tensor<2,dim,ADNumberType>
  //				  F_AD =
  // Physics::Elasticity::Kinematics::F(solution_grads_u[q_point]);
  // ADNumberType det_F_AD = determinant(F_AD); 				  const double det_F =
  // Tensor<0,dim,double>(det_F_AD); 				  det_F_mpi.push_back (det_F);
  //
  //				  const std::vector<std::shared_ptr<const
  // PointHistory<dim,ADNumberType>>> 				  lqph =
  // quadrature_point_history.get_data(cell); 				  Assert(lqph.size() ==
  // n_q_points, ExcInternalError());
  //
  //				  double JxW = fe_values_ref.JxW(q_point);
  //
  //				  //Volumes
  //				  sum_vol_current_mpi  += det_F * JxW;
  //				  sum_vol_reference_mpi += JxW;
  //				  //sum_solid_vol_mpi += parameters.solid_vol_frac * JxW * det_F;
  //				  sum_solid_vol_mpi += parameters.solid_vol_frac * JxW;
  //
  //				  //Seepage velocity
  //				  const Tensor<2,dim,ADNumberType> F_inv = invert(F_AD);
  //				  const Tensor<1,dim,ADNumberType>
  //				  grad_p_fluid_AD =  solution_grads_p_fluid_AD[q_point]*F_inv;
  //				  const Tensor<1,dim,ADNumberType> seepage_vel_AD
  //				  = lqph[q_point]->get_seepage_velocity_current(F_AD,
  // grad_p_fluid_AD);
  //
  //				  Tensor<1,dim> seepage;
  //				  for (unsigned int i=0; i<dim; ++i)
  //					  seepage[i] = Tensor<0,dim,double>(seepage_vel_AD[i]);
  //
  //				  /*//test
  //                      SymmetricTensor<2,dim> sigma_E_ext_func;
  //                      const SymmetricTensor<2,dim,ADNumberType>
  //                      sigma_E_ext_func_AD =
  //                      lqph[q_point]->get_Cauchy_E_ext_func(F_AD);
  //
  //                      double det_F_converged =
  //                      lqph[q_point]->get_converged_det_F();
  //
  //                      for (unsigned int i=0; i<dim; ++i)
  //                      	for (unsigned int j=0; j<dim; ++j) {
  //                      		sigma_E_ext_func[i][j] =
  //                      Tensor<0,dim,double>(sigma_E_ext_func_AD[i][j]);
  //                      	}
  //
  //                      const Point<dim> gauss_coord2 =
  //                      fe_values_ref.quadrature_point(q_point); std::ofstream
  //                      sigma_ext_func_cells;
  //                      sigma_ext_func_cells.open("sigma_ext_func_cells",
  //                      std::ofstream::app); sigma_ext_func_cells <<
  //                      std::setprecision(8) << std::scientific;
  //                      sigma_ext_func_cells << std::setw(16) <<
  //                      this->time->get_current() << ","
  //                      		<< std::setw(16) << gauss_coord2[0] << ","
  //      						<< std::setw(16) << gauss_coord2[1] << ","
  //      						<< std::setw(16) << gauss_coord2[2] << ","
  //      						<< std::setw(16) << JxW << ","
  //      						<< std::setw(16) << det_F << ","
  //      						<< std::setw(16) << det_F_converged << ","
  //      						<< std::setw(16) << sigma_E_ext_func[0][0] << ","
  //      						<< std::setw(16) << sigma_E_ext_func[1][1] << ","
  //      						<< std::setw(16) << sigma_E_ext_func[2][2] << std::endl;
  //                      sigma_ext_func_cells.close();
  //
  //                      //if (seepage[2]>0)
  //                      //	std::cout << seepage[2] << " cell loop" <<
  //                      std::endl;
  //
  //                      //const Point<dim> gauss_coord =
  //                      fe_values_ref.quadrature_point(q_point);
  //                      //if (gauss_coord[2] < 0.1) {
  //                      //	seepage_vec_mpi.push_back (seepage[2]);
  //                      	//std::cout << seepage[2] << std::endl;
  //                      //}*/
  //
  //				  //Dissipations
  //				  const double porous_dissipation =
  //						  lqph[q_point]->get_porous_dissipation(F_AD, grad_p_fluid_AD);
  //				  sum_porous_dissipation_mpi += porous_dissipation * det_F * JxW;
  //
  //				  const double viscous_dissipation =
  // lqph[q_point]->get_viscous_dissipation(); sum_viscous_dissipation_mpi
  // += viscous_dissipation * det_F * JxW;
  //
  //				  //---------------------------------------------------------------
  //			  } //end gauss point loop
  //
  //			  // Compute reaction force on load boundary & total fluid flow across
  //			  // drained boundary.
  //			  // Define a local instance of FEFaceValues to compute values
  // required
  //			  // to calculate reaction force
  //			  const UpdateFlags uf_face( update_values | update_gradients |
  //					  update_normal_vectors | update_JxW_values | update_quadrature_points);
  //			  FEFaceValues<dim> fe_face_values_ref(mapping, fe, qf_face, uf_face);
  //
  //			  //start face loop
  //			  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell;
  //++face)
  //			  {
  //				  //Reaction force
  //				  if (cell->face(face)->at_boundary() == true &&
  //						  cell->face(face)->boundary_id() ==
  // get_reaction_boundary_id_for_output() )
  //				  {
  //					  fe_face_values_ref.reinit(cell, face);
  //
  //					  //Get displacement gradients for current face
  //					  std::vector<Tensor<2,dim> > solution_grads_u_f(n_q_points_f);
  //					  fe_face_values_ref[u_fe].get_function_gradients
  //					  (solution_total,
  //							  solution_grads_u_f);
  //
  //					  //Get pressure for current element
  //					  std::vector< double >
  // solution_values_p_fluid_total_f(n_q_points_f);
  //					  fe_face_values_ref[p_fluid_fe].get_function_values
  //					  (solution_total,
  //							  solution_values_p_fluid_total_f);
  //
  //					  //start gauss points on faces loop
  //					  for (unsigned int f_q_point=0; f_q_point<n_q_points_f;
  //++f_q_point)
  //					  {
  //						  const Tensor<1,dim> &N =
  // fe_face_values_ref.normal_vector(f_q_point); 						  const double
  // JxW_f = fe_face_values_ref.JxW(f_q_point);
  //
  //						  //Compute deformation gradient from displacements gradient
  //						  //(present configuration)
  //						  const Tensor<2,dim,ADNumberType> F_AD =
  //								  Physics::Elasticity::Kinematics::F(solution_grads_u_f[f_q_point]);
  //						  ADNumberType det_F_AD = determinant(F_AD);
  //						  double det_F = Tensor<0,dim,double>(det_F_AD);
  //
  //						  const std::vector<std::shared_ptr<const
  // PointHistory<dim,ADNumberType>>> 						  lqph =
  // quadrature_point_history.get_data(cell); 						  Assert(lqph.size()
  // == n_q_points, ExcInternalError());
  //
  //						  const double p_fluid =
  // solution_values_p_fluid_total[f_q_point];
  //
  //						  //Cauchy stress
  //						  static const SymmetricTensor<2,dim,double> I
  //(Physics::Elasticity::StandardTensors<dim>::I); SymmetricTensor<2,dim>
  // sigma_E; 						  SymmetricTensor<2,dim> sigma_E_base;
  // SymmetricTensor<2,dim> sigma_E_ext_func; 						  const
  // SymmetricTensor<2,dim,ADNumberType> sigma_E_AD =
  // lqph[f_q_point]->get_Cauchy_E(F_AD); 						  const
  // SymmetricTensor<2,dim,ADNumberType> sigma_E_base_AD =
  // lqph[f_q_point]->get_Cauchy_E_base(F_AD); 						  const
  // SymmetricTensor<2,dim,ADNumberType> sigma_E_ext_func_AD =
  // lqph[f_q_point]->get_Cauchy_E_ext_func(F_AD);
  //
  //						  double det_F_converged =
  // lqph[f_q_point]->get_converged_det_F();
  //
  //						  for (unsigned int i=0; i<dim; ++i)
  //							  for (unsigned int j=0; j<dim; ++j) {
  //								  sigma_E[i][j] = Tensor<0,dim,double>(sigma_E_AD[i][j]);
  //								  sigma_E_base[i][j] =
  // Tensor<0,dim,double>(sigma_E_base_AD[i][j]); sigma_E_ext_func[i][j] =
  // Tensor<0,dim,double>(sigma_E_ext_func_AD[i][j]);
  //							  }
  //
  //						  SymmetricTensor<2,dim> sigma_fluid_vol(I);
  //						  sigma_fluid_vol *= -1.0*p_fluid;
  //						  const SymmetricTensor<2,dim> sigma = sigma_E+sigma_fluid_vol;
  //						  sum_reaction_mpi += sigma * N * JxW_f;
  //						  sum_reaction_pressure_mpi += sigma_fluid_vol * N * JxW_f;
  //						  sum_reaction_extra_mpi += sigma_E * N * JxW_f;
  //						  sum_reaction_extra_base_mpi += sigma_E_base * N * JxW_f;
  //						  sum_reaction_extra_ext_func_mpi += sigma_E_ext_func * N *
  // JxW_f;
  //
  //
  //						  /*const Point<dim> gauss_coord2 =
  // fe_face_values_ref.quadrature_point(f_q_point);
  //                              std::ofstream
  //                              sigma_ext_func_reaction_force_boundary;
  //                              sigma_ext_func_reaction_force_boundary.open("sigma_ext_func_reaction_force_boundary",
  //                              std::ofstream::app);
  //                              sigma_ext_func_reaction_force_boundary <<
  //                              std::setprecision(8) << std::scientific;
  //                              sigma_ext_func_reaction_force_boundary <<
  //                              std::setw(16) << this->time->get_current() <<
  //                              ","
  //                              		<< std::setw(16) << gauss_coord2[0] << ","
  //      								<< std::setw(16) << gauss_coord2[1] << ","
  //      								<< std::setw(16) << gauss_coord2[2] << ","
  //      								<< std::setw(16) << JxW_f << ","
  //      								<< std::setw(16) << det_F << ","
  //      								<< std::setw(16) << det_F_converged << ","
  //                              		<< std::setw(16) << sigma_E_ext_func[0][0]
  //                              << ","
  //                              		<< std::setw(16) << sigma_E_ext_func[1][1]
  //                              << ","
  //      								<< std::setw(16) << sigma_E_ext_func[2][2] <<
  //      std::endl;
  //                              sigma_ext_func_reaction_force_boundary.close();*/
  //
  //						  //Transform components of Cauchy stresses into cylindrical
  // coordinates for torque
  //						  //evaluation under torsional shear loading
  //
  //						  //Obtain spherical coordinates from current Gauss Point
  //						  const std::array<double,dim> gauss_sph =
  // GeometricUtilities::Coordinates::to_spherical(fe_face_values_ref.quadrature_point(f_q_point));
  //						  const Point<dim> gauss_coord =
  // fe_face_values_ref.quadrature_point(f_q_point);
  //						  //Compute azimuth angle and radius
  //						  const double theta = gauss_sph[1];
  //						  //std::cout << theta << std::endl;
  //						  const double r = sqrt(pow(gauss_coord[0],2) +
  // pow(gauss_coord[1],2));
  //						  //Define a new 2nd order tensor to store the cylindrical
  // coordinates of sigma_E 						  SymmetricTensor<2,dim> sigma_E_cyl;
  //
  //						  //Compute the components of sigma_E_cyl associated with the
  // corresponding cylindrical
  //						  //coordinate system from sigma_E
  //						  sigma_E_cyl[0][0] = sigma_E[0][0]*pow(cos(theta),2) +
  // sigma_E[1][1]*pow(sin(theta),2) + 2*sigma_E[0][1]*sin(theta)*cos(theta);
  ////sigma_r_r 						  sigma_E_cyl[1][1] =
  /// sigma_E[0][0]*pow(sin(theta),2)
  ///+
  // sigma_E[1][1]*pow(cos(theta),2) - 2*sigma_E[0][1]*sin(theta)*cos(theta);
  ////sigma_theta_theta 						  sigma_E_cyl[0][1] =
  //(sigma_E[1][1]-sigma_E[0][0])*sin(theta)*cos(theta) +
  // sigma_E[0][1]*(pow(cos(theta),2)-pow(sin(theta),2)); //sigma_r_theta
  //						  sigma_E_cyl[0][2] = sigma_E[0][2]*cos(theta) +
  // sigma_E[1][2]*sin(theta); //sigma_r_z 						  sigma_E_cyl[1][2] =
  // sigma_E[1][2]*cos(theta) - sigma_E[0][2]*sin(theta); //sigma_theta_z
  //						  sigma_E_cyl[2][2] = sigma_E[2][2]; //sigma_zz
  //						  sigma_E_cyl[1][0] = sigma_E_cyl[0][1]; //sigma_theta_r
  //						  sigma_E_cyl[2][0] = sigma_E_cyl[0][2]; //sigma_z_r
  //						  sigma_E_cyl[2][1] = sigma_E_cyl[1][2]; //sigma_z_theta
  //
  //						  //Analogeously define and compute cylindrical components of
  // sigma_fluid_vol 						  SymmetricTensor<2,dim> sigma_fluid_vol_cyl;
  //						  sigma_fluid_vol_cyl[0][0] =
  // sigma_fluid_vol[0][0]*pow(cos(theta),2) +
  // sigma_fluid_vol[1][1]*pow(sin(theta),2) +
  // 2*sigma_fluid_vol[0][1]*sin(theta)*cos(theta); //sigma_r_r
  //						  sigma_fluid_vol_cyl[1][1] =
  // sigma_fluid_vol[0][0]*pow(sin(theta),2) +
  // sigma_fluid_vol[1][1]*pow(cos(theta),2) -
  // 2*sigma_fluid_vol[0][1]*sin(theta)*cos(theta); //sigma_theta_theta
  //						  sigma_fluid_vol_cyl[0][1] =
  //(sigma_fluid_vol[1][1]-sigma_fluid_vol[0][0])*sin(theta)*cos(theta) +
  // sigma_fluid_vol[0][1]*(pow(cos(theta),2)-pow(sin(theta),2));
  // //sigma_r_theta 						  sigma_fluid_vol_cyl[0][2] =
  // sigma_fluid_vol[0][2]*cos(theta) + sigma_fluid_vol[1][2]*sin(theta);
  // //sigma_r_z 						  sigma_fluid_vol_cyl[1][2] =
  // sigma_fluid_vol[1][2]*cos(theta) - sigma_fluid_vol[0][2]*sin(theta);
  ////sigma_theta_z 						  sigma_fluid_vol_cyl[2][2] =
  /// sigma_fluid_vol[2][2]; /sigma_zz 						  sigma_fluid_vol_cyl[1][0] =
  /// sigma_fluid_vol_cyl[0][1]; /sigma_theta_r sigma_fluid_vol_cyl[2][0] =
  /// sigma_fluid_vol_cyl[0][2]; /sigma_z_r sigma_fluid_vol_cyl[2][1] =
  /// sigma_fluid_vol_cyl[1][2]; /sigma_z_theta
  //
  //						  //Add solid and fluid component to get total Cauchy stress
  //						  const SymmetricTensor<2,dim> sigma_cyl = sigma_E_cyl +
  // sigma_fluid_vol_cyl;
  //
  //						  //Compute torque contribution from z_theta component, the
  // according weighted area JxW_f
  //						  //and the distance r to center of rotation
  //						  sum_torque_mpi += sigma_cyl[2][1] * r * JxW_f;
  //					  }//end gauss points on faces loop
  //				  }
  //
  //
  //
  //				  //Fluid flow
  //				  if (cell->face(face)->at_boundary() == true &&
  //						  (cell->face(face)->boundary_id() ==
  //								  get_drained_boundary_id_for_output().first ||
  //								  cell->face(face)->boundary_id() ==
  //										  get_drained_boundary_id_for_output().second ) )
  //				  {
  //					  fe_face_values_ref.reinit(cell, face);
  //
  //					  //Get displacement gradients for current face
  //					  std::vector<Tensor<2,dim>> solution_grads_u_f(n_q_points_f);
  //					  fe_face_values_ref[u_fe].get_function_gradients
  //					  (solution_total,
  //							  solution_grads_u_f);
  //
  //					  //Get pressure gradients for current face
  //					  std::vector<Tensor<1,dim>> solution_grads_p_f(n_q_points_f);
  //					  fe_face_values_ref[p_fluid_fe].get_function_gradients
  //					  (solution_total,
  //							  solution_grads_p_f);
  //
  //					  //start gauss points on faces loop
  //					  for (unsigned int f_q_point=0; f_q_point<n_q_points_f;
  //++f_q_point)
  //					  {
  //						  const Tensor<1,dim> &N =
  // fe_face_values_ref.normal_vector(f_q_point); 						  const double
  // JxW_f = fe_face_values_ref.JxW(f_q_point);
  //
  //						  //Deformation gradient and inverse from displacements gradient
  //						  //(present configuration)
  //						  const Tensor<2,dim,ADNumberType> F_AD
  //						  =
  // Physics::Elasticity::Kinematics::F(solution_grads_u_f[f_q_point]);
  //
  //						  const Tensor<2,dim,ADNumberType> F_inv_AD = invert(F_AD);
  //						  ADNumberType det_F_AD = determinant(F_AD);
  //						  const Tensor<2,dim,ADNumberType> F_inv_AD_trans =
  // transpose(F_inv_AD); //added-1.10.21
  //
  //						  const std::vector<std::shared_ptr<const
  // PointHistory<dim,ADNumberType>>> 						  lqph =
  // quadrature_point_history.get_data(cell); 						  Assert(lqph.size()
  // == n_q_points, ExcInternalError());
  //
  //						  //Seepage velocity
  //						  Tensor<1,dim> seepage;
  //						  //double det_F = Tensor<0,dim,double>(det_F_AD);
  //						  const Tensor<1,dim,ADNumberType> grad_p
  //						  = solution_grads_p_f[f_q_point]*F_inv_AD;
  //						  const Tensor<1,dim,ADNumberType> seepage_AD
  //						  = lqph[f_q_point]->get_seepage_velocity_current(F_AD, grad_p);
  //
  //						  for (unsigned int i=0; i<dim; ++i)
  //							  seepage[i] = Tensor<0,dim,double>(seepage_AD[i]);
  //
  //						  //sum_total_flow_mpi += seepage * N * JxW_f;
  //						  //if (seepage[2]>0)
  //						  //	std::cout << seepage[2] << " face loop" << std::endl;
  //						  //sum_total_flow_mpi += (seepage/det_F) * N * JxW_f;
  //						  const Tensor<1,dim> temp1 =
  // contract<0,0>(seepage,F_inv_AD_trans); 						  sum_total_flow_mpi +=
  // temp1
  // * N * JxW_f; //added-1.10.21
  //					  }//end gauss points on faces loop
  //				  }
  //
  //
  //				  // Minimal Jacobian
  //				  if (cell->face(face)->at_boundary() == true) {
  //					  fe_face_values_ref.reinit(cell, face);
  //
  //					  //Get displacement gradients for current face
  //					  std::vector<Tensor<2,dim> > solution_grads_u_f(n_q_points_f);
  //					  fe_face_values_ref[u_fe].get_function_gradients(solution_total,
  // solution_grads_u_f);
  //
  //					  //start Gauss points loop on faces
  //					  for (unsigned int f_q_point=0; f_q_point<n_q_points_f;
  //++f_q_point) {
  //						  //Compute deformation gradient from displacements gradient and
  // its Jacobian 						  const Tensor<2,dim,ADNumberType> F_AD =
  // Physics::Elasticity::Kinematics::F(solution_grads_u_f[f_q_point]);
  //						  ADNumberType det_F_AD = determinant(F_AD);
  //						  det_F_mpi.push_back(Tensor<0,dim,double>(det_F_AD));
  //
  //						  /*//test
  //                      		const std::vector<std::shared_ptr<const
  //                      PointHistory<dim,ADNumberType>>> lqph =
  //                      quadrature_point_history.get_data(cell);
  //                      		Assert(lqph.size() == n_q_points,
  //                      ExcInternalError());
  //
  //                              SymmetricTensor<2,dim> sigma_E_ext_func;
  //                              const SymmetricTensor<2,dim,ADNumberType>
  //                              sigma_E_ext_func_AD =
  //                              lqph[f_q_point]->get_Cauchy_E_ext_func(F_AD);
  //
  //                              double det_F_converged =
  //                              lqph[f_q_point]->get_converged_det_F(); double
  //                              det_F = Tensor<0,dim,double>(det_F_AD); const
  //                              double JxW_f =
  //                              fe_face_values_ref.JxW(f_q_point);
  //
  //                              for (unsigned int i=0; i<dim; ++i)
  //                              	for (unsigned int j=0; j<dim; ++j) {
  //                              		sigma_E_ext_func[i][j] =
  //                              Tensor<0,dim,double>(sigma_E_ext_func_AD[i][j]);
  //                              	}
  //
  //                              const Point<dim> gauss_coord2 =
  //                              fe_face_values_ref.quadrature_point(f_q_point);
  //                              std::ofstream sigma_ext_func_faces;
  //                              sigma_ext_func_faces.open("sigma_ext_func_faces",
  //                              std::ofstream::app); sigma_ext_func_faces <<
  //                              std::setprecision(8) << std::scientific;
  //                              sigma_ext_func_faces << std::setw(16) <<
  //                              this->time->get_current() << ","
  //                              		<< std::setw(16) << gauss_coord2[0] << ","
  //              						<< std::setw(16) << gauss_coord2[1] << ","
  //              						<< std::setw(16) << gauss_coord2[2] << ","
  //              						<< std::setw(16) << JxW_f << ","
  //              						<< std::setw(16) << det_F << ","
  //              						<< std::setw(16) << det_F_converged << ","
  //              						<< std::setw(16) << sigma_E_ext_func[0][0] << ","
  //              						<< std::setw(16) << sigma_E_ext_func[1][1] << ","
  //              						<< std::setw(16) << sigma_E_ext_func[2][2] <<
  //              std::endl;
  //                              sigma_ext_func_faces.close();*/
  //					  }
  //				  }
  //			  }//end face loop
  //		  }//end cell loop
  //
  //		  //Sum the results from different MPI process and then add to the
  // reaction_force vector
  //		  //In theory, the solution on each surface (each cell) only exists in
  // one MPI process
  //		  //so, we add all MPI process, one will have the solution and the
  // others will be zero 		  for (unsigned int d=0; d<dim; ++d)
  //		  {
  //			  reaction_force[d] = Utilities::MPI::sum(sum_reaction_mpi[d],
  // mpi_communicator); 			  reaction_force_pressure[d] =
  // Utilities::MPI::sum(sum_reaction_pressure_mpi[d], mpi_communicator);
  //			  reaction_force_extra[d] =
  // Utilities::MPI::sum(sum_reaction_extra_mpi[d], mpi_communicator);
  //			  reaction_force_extra_base[d] =
  // Utilities::MPI::sum(sum_reaction_extra_base_mpi[d], mpi_communicator);
  //			  reaction_force_extra_ext_func[d] =
  // Utilities::MPI::sum(sum_reaction_extra_ext_func_mpi[d], mpi_communicator);
  //		  }
  //
  //		  //Same for total fluid flow, and for porous and viscous dissipations
  //		  total_fluid_flow = Utilities::MPI::sum(sum_total_flow_mpi,
  // mpi_communicator); 		  total_porous_dissipation =
  // Utilities::MPI::sum(sum_porous_dissipation_mpi, mpi_communicator);
  //		  total_viscous_dissipation =
  // Utilities::MPI::sum(sum_viscous_dissipation_mpi, mpi_communicator);
  //		  total_solid_vol = Utilities::MPI::sum(sum_solid_vol_mpi,
  // mpi_communicator); 		  total_vol_current =
  // Utilities::MPI::sum(sum_vol_current_mpi, mpi_communicator);
  //		  total_vol_reference = Utilities::MPI::sum(sum_vol_reference_mpi,
  // mpi_communicator); 		  reaction_torque =
  // Utilities::MPI::sum(sum_torque_mpi, mpi_communicator);
  //
  //		  det_F_min_mpi = *std::min_element(det_F_mpi.begin(),det_F_mpi.end());
  //		  det_F_min = Utilities::MPI::min(det_F_min_mpi, mpi_communicator);
  //
  //		  //if (seepage_vec_mpi.size()>1) {
  //			  //	seepage_vec_mean_mpi = std::accumulate(seepage_vec_mpi.begin(),
  // seepage_vec_mpi.end(), 0.0) / seepage_vec_mpi.size();
  //		  //	std::cout << seepage_vec_mean_mpi << std::endl;
  //		  //	seepage_vec_mean = seepage_vec_mean_mpi;
  //		  //}
  //		  // seepage_vec_mean = Utilities::MPI::sum(seepage_vec_mean_mpi,
  // mpi_communicator) / 6;
  //
  //		  //  Extract solution for tracked vectors
  //		  // Copying an MPI::BlockVector into MPI::Vector is not possible,
  //		  // so we copy each block of MPI::BlockVector into an MPI::Vector
  //		  // And then we copy the MPI::Vector into "normal" Vectors
  //		  TrilinosWrappers::MPI::Vector
  // solution_vector_u_MPI(solution_total.block(u_block));
  //		  TrilinosWrappers::MPI::Vector
  // solution_vector_p_MPI(solution_total.block(p_fluid_block)); Vector<double>
  // solution_u_vector(solution_vector_u_MPI); 		  Vector<double>
  // solution_p_vector(solution_vector_p_MPI);
  //
  //		  if (this_mpi_process == 0)
  //		  {
  //			  //Append the pressure solution vector to the displacement solution
  // vector,
  //			  //creating a single solution vector equivalent to the original
  // BlockVector
  //			  //so FEFieldFunction will work with the dof_handler_ref.
  //			  Vector<double> solution_vector(solution_p_vector.size()
  //					  +solution_u_vector.size());
  //
  //			  for (unsigned int d=0; d<(solution_u_vector.size()); ++d)
  //				  solution_vector[d] = solution_u_vector[d];
  //
  //			  for (unsigned int d=0; d<(solution_p_vector.size()); ++d)
  //				  solution_vector[solution_u_vector.size()+d] =
  // solution_p_vector[d];
  //
  //			  Functions::FEFieldFunction<dim,DoFHandler<dim>,Vector<double>>
  //			  find_solution(dof_handler_ref, solution_vector);
  //
  //			  for (unsigned int p=0; p<tracked_vertices_IN.size(); ++p)
  //			  {
  //				  Vector<double> update(dim+1);
  //				  Point<dim> pt_ref;
  //
  //				  pt_ref[0]= tracked_vertices_IN[p][0];
  //				  pt_ref[1]= tracked_vertices_IN[p][1];
  //				  pt_ref[2]= tracked_vertices_IN[p][2];
  //
  //				  find_solution.vector_value(pt_ref, update);
  //
  //				  for (unsigned int d=0; d<(dim+1); ++d)
  //				  {
  //					  //For values close to zero, set to 0.0
  //					  if (abs(update[d])<1.5*parameters.tol_u)
  //						  update[d] = 0.0;
  //					  solution_vertices[p][d] = update[d];
  //				  }
  //			  }
  //			  // Write the results to the plotting file.
  //			  // Add two blank lines between cycles in the cyclic loading examples
  // so GNUPLOT can detect each cycle as a different block 			  if
  //((parameters.geom_type == "Budday_cube_tension_compression_fully_fixed")||
  //					  (parameters.geom_type == "Budday_cube_tension_compression")||
  //					  (parameters.geom_type == "Budday_cube_shear_fully_fixed")){
  //				  const double delta_time = time->get_delta_t();
  //				  const double end_time   = time->get_end();
  //
  //				  // This was previously called from parameters.
  //				  // Current time is passed into the function, maybe it can be
  // called from time-> too?
  //
  //				  if (( (parameters.geom_type ==
  //"Budday_cube_tension_compression_fully_fixed")|| (parameters.geom_type ==
  //"Budday_cube_tension_compression")|| 						  (parameters.geom_type ==
  //"Budday_cube_shear_fully_fixed")                ) && 						  (
  //(abs(current_time - end_time/3.)   <0.9*delta_time)||
  // (abs(current_time
  //- 2.*end_time/3.)<0.9*delta_time)   ) &&
  // parameters.num_cycle_sets == 1 )
  //				  {
  //					  plotpointfile << std::endl<< std::endl;
  //				  }
  //				  if (( (parameters.geom_type ==
  //"Budday_cube_tension_compression_fully_fixed")|| (parameters.geom_type ==
  //"Budday_cube_tension_compression")|| 						  (parameters.geom_type ==
  //"Budday_cube_shear_fully_fixed")             ) && 						  (
  //(abs(current_time - end_time/9.)   <0.9*delta_time)||
  // (abs(current_time
  //- 2.*end_time/9.)<0.9*delta_time)|| 								  (abs(current_time
  //- 3.*end_time/9.)<0.9*delta_time)|| 								  (abs(current_time
  //- 5.*end_time/9.)<0.9*delta_time)|| 								  (abs(current_time
  //- 7.*end_time/9.)<0.9*delta_time) ) && parameters.num_cycle_sets == 2 )
  //				  {
  //					  plotpointfile << std::endl<< std::endl;
  //				  }
  //			  }
  //
  //			  plotpointfile <<  std::setprecision(6) << std::scientific;
  //			  plotpointfile << std::setw(16) << current_time        << ","
  //					  << std::setw(15) << total_vol_reference << ","
  //					  << std::setw(15) << total_vol_current   << ","
  //					  << std::setw(15) << total_solid_vol     << ",";
  //
  //			  if (current_time == 0.0)
  //			  {
  //				  for (unsigned int p=0; p<tracked_vertices_IN.size(); ++p)
  //				  {
  //					  for (unsigned int d=0; d<dim; ++d)
  //						  plotpointfile << std::setw(15) << 0.0 << ",";
  //
  //					  plotpointfile << std::setw(15) << parameters.drained_pressure <<
  //",";
  //				  }
  //				  for (unsigned int d=0; d<(3*dim+2); ++d)
  //					  plotpointfile << std::setw(15) << 0.0 << ",";
  //
  //				  plotpointfile << std::setw(15) << 0.0 << ","
  //						  << std::setw(15) << 0.0 << ",";
  //
  //				  for (unsigned int d=0; d<(2*dim); ++d)
  //					  plotpointfile << std::setw(15) << 0.0 << ",";
  //
  //				  plotpointfile << std::setw(15) << 1.0 << ",";
  //			  }
  //			  else
  //			  {
  //				  for (unsigned int p=0; p<tracked_vertices_IN.size(); ++p)
  //					  for (unsigned int d=0; d<(dim+1); ++d)
  //						  plotpointfile << std::setw(15) << solution_vertices[p][d]<<
  //",";
  //
  //				  if (parameters.geom_type ==
  //"brain_rheometer_cyclic_tension_compression_exp_quarter" ||
  // parameters.geom_type == "hydro_nano_graz_compression_exp_relax"){ for
  //(unsigned int d=0; d<dim; ++d) 						  plotpointfile << std::setw(15)
  //<<
  // reaction_force[d]*(4e-6) << ","; 				  } else { 					  for
  // (unsigned int d=0; d<dim; ++d) 						  plotpointfile << std::setw(15)
  // << reaction_force[d] <<
  //",";
  //				  }
  //
  //				  for (unsigned int d=0; d<dim; ++d)
  //					  plotpointfile << std::setw(15) << reaction_force_pressure[d] <<
  //",";
  //
  //				  for (unsigned int d=0; d<dim; ++d)
  //					  plotpointfile << std::setw(15) << reaction_force_extra[d] <<
  //",";
  //
  //				  plotpointfile << std::setw(15) << total_fluid_flow << ","
  //						  //<< std::setw(15) << seepage_vec_mean << ","
  //						  << std::setw(15) << total_porous_dissipation << ","
  //						  << std::setw(15) << total_viscous_dissipation << ","
  //						  << std::setw(15) << reaction_torque << ",";
  //
  //				  for (unsigned int d=0; d<dim; ++d)
  //					  plotpointfile << std::setw(15) << reaction_force_extra_base[d]
  //<< ","; 				  for (unsigned int d=0; d<dim; ++d) plotpointfile <<
  // std::setw(15) << reaction_force_extra_ext_func[d] << ",";
  //
  //				  plotpointfile << std::setw(15) << det_F_min << ",";
  //			  }
  //			  plotpointfile << std::endl;
  //		  }
  //			  }


  // Header for console output file
  template <int dim>
  void
  Solid<dim>::print_console_file_header(std::ofstream &outputfile) const
  {
    outputfile
      << "/*-----------------------------------------------------------------------------------------";
    outputfile
      << "\n\n  Poro-viscoelastic formulation to solve nonlinear solid mechanics problems using deal.ii";
    outputfile
      << "\n\n  Problem setup by E Comellas and J-P Pelteret, University of Erlangen-Nuremberg, 2018";
    outputfile
      << "\n\n/*-----------------------------------------------------------------------------------------";
    outputfile << "\n\nCONSOLE OUTPUT: \n\n";
  }

  // Header for plotting output file
  template <int dim>
  void
  Solid<dim>::print_plot_file_header(std::vector<Point<dim>> &tracked_vertices,
                                     std::ofstream &plotpointfile) const
  {
    plotpointfile
      << "#\n# *** Solution history for tracked vertices -- DOF: 0 = Ux,  1 = Uy,  2 = Uz,  3 = P ***"
      << std::endl;

    for (unsigned int p = 0; p < tracked_vertices.size(); ++p)
      {
        plotpointfile << "#        Point " << p << " coordinates:  ";
        for (unsigned int d = 0; d < dim; ++d)
          {
            plotpointfile << tracked_vertices[p][d];
            if (!((p == tracked_vertices.size() - 1) && (d == dim - 1)))
              plotpointfile << ",        ";
          }
        plotpointfile << std::endl;
      }
    plotpointfile
      << "#    The reaction force is the integral over the loaded surfaces in the "
      << "undeformed configuration of the Cauchy stress times the normal surface unit vector.\n"
      << "#    reac(p) corresponds to the volumetric part of the Cauchy stress due to the pore fluid pressure"
      << " and reac(E) corresponds to the extra part of the Cauchy stress due to the solid contribution."
      << std::endl
      << "#    The fluid flow is the integral over the drained surfaces in the "
      << "undeformed configuration of the seepage velocity times the normal surface unit vector."
      << std::endl
      << "# Column number:" << std::endl
      << "#";

    unsigned int columns = 33;
    for (unsigned int d = 1; d < columns; ++d)
      plotpointfile << std::setw(15) << d << ",";

    plotpointfile << std::setw(15) << columns << std::endl
                  << "#" << std::right << std::setw(16) << "Time," << std::right
                  << std::setw(16) << "ref vol," << std::right << std::setw(16)
                  << "def vol," << std::right << std::setw(16) << "solid vol,";

    for (unsigned int p = 0; p < tracked_vertices.size(); ++p)
      for (unsigned int d = 0; d < (dim + 1); ++d)
        plotpointfile << std::right << std::setw(11) << "P" << p << "[" << d
                      << "],";

    for (unsigned int d = 0; d < dim; ++d)
      plotpointfile << std::right << std::setw(13) << "reaction [" << d << "],";

    for (unsigned int d = 0; d < dim; ++d)
      plotpointfile << std::right << std::setw(13) << "reac(p) [" << d << "],";

    for (unsigned int d = 0; d < dim; ++d)
      plotpointfile << std::right << std::setw(13) << "reac(E) [" << d << "],";

    plotpointfile << std::right << std::setw(16) << "fluid flow," << std::right
                  << std::setw(16) << "porous dissip," << std::right
                  << std::setw(16) << "viscous dissip," << std::right
                  << std::setw(16) << "torque,";

    for (unsigned int d = 0; d < dim; ++d)
      plotpointfile << std::right << std::setw(13) << "base(E) [" << d << "],";

    for (unsigned int d = 0; d < dim; ++d)
      plotpointfile << std::right << std::setw(13) << "ext_func(E) [" << d
                    << "],";

    plotpointfile << std::right << std::setw(16) << "J_min_cell,";
    plotpointfile << std::right << std::setw(16) << "J_min_face,";
    plotpointfile << std::endl;
  }

  // Footer for console output file
  template <int dim>
  void
  Solid<dim>::print_console_file_footer(std::ofstream &outputfile) const
  {
    // Copy "parameters" file at end of output file.
    std::ifstream infile(parameters.output_directory + "parameters.prm");
    std::string   content = "";
    // int           i;

    while (infile.eof() != true)
      // for (i = 0; infile.eof() != true; i++)
      {
        char aux = infile.get();
        content += aux;
        if (aux == '\n')
          content += '#';
      }

    // i--;
    content.erase(content.end() - 1);
    infile.close();

    outputfile << "\n\n\n\n PARAMETERS FILE USED IN THIS COMPUTATION: \n#"
               << std::endl
               << content;
  }

  // Footer for plotting output file
  template <int dim>
  void
  Solid<dim>::print_plot_file_footer(std::ofstream &plotpointfile) const
  {
    // Copy "parameters" file at end of output file.
    std::ifstream infile(parameters.output_directory + "parameters.prm");
    std::cout << "here11" << std::endl;
    std::string content = "";
    // int         i;
    while (infile.eof() != true)
      // for (i = 0; infile.eof() != true; i++)
      {
        char aux = infile.get();
        content += aux;
        if (aux == '\n')
          content += '#';
      }
    std::cout << "here12" << std::endl;
    // i--;
    content.erase(content.end() - 1);
    std::cout << "here13" << std::endl;
    infile.close();
    std::cout << "here14" << std::endl;

    plotpointfile << "#" << std::endl
                  << "#" << std::endl
                  << "# PARAMETERS FILE USED IN THIS COMPUTATION:" << std::endl
                  << "#" << std::endl
                  << content;
  }


  // @sect3{Verification examples from Ehlers and Eipper 1999}
  // We group the definition of the geometry, boundary and loading conditions
  // specific to the verification examples from Ehlers and Eipper 1999 into
  // specific classes.

  //@sect4{Base class: Tube geometry and boundary conditions}
  template <int dim>
  class VerificationEhlers1999TubeBase : public Solid<dim>
  {
  public:
    VerificationEhlers1999TubeBase(const Parameters::AllParameters &parameters)
      : Solid<dim>(parameters)
    {}

    virtual ~VerificationEhlers1999TubeBase()
    {}

  private:
    virtual void
    make_grid() override
    {
      GridGenerator::cylinder(this->triangulation, 0.1, 0.5);

      const double rot_angle = 3.0 * numbers::PI / 2.0;
      GridTools::rotate(rot_angle, 1, this->triangulation);

      this->triangulation.reset_manifold(0);
      static const CylindricalManifold<dim> manifold_description_3d(2);
      this->triangulation.set_manifold(0, manifold_description_3d);
      GridTools::scale(this->parameters.scale, this->triangulation);
      this->triangulation.refine_global(
        std::max(1U, this->parameters.global_refinement));
      this->triangulation.reset_manifold(0);
    }

    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.0 * this->parameters.scale;
      tracked_vertices[0][1] = 0.0 * this->parameters.scale;
      tracked_vertices[0][2] = 0.5 * this->parameters.scale;

      tracked_vertices[1][0] = 0.0 * this->parameters.scale;
      tracked_vertices[1][1] = 0.0 * this->parameters.scale;
      tracked_vertices[1][2] = -0.5 * this->parameters.scale;
    }

    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        0,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement)));

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        1,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));
    }

    virtual double
    get_prescribed_fluid_flow(const types::boundary_id &boundary_id,
                              const Point<dim>         &pt) const override
    {
      (void)pt;
      (void)boundary_id;
      return 0.0;
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 2;
    }

    virtual std::pair<types::boundary_id, types::boundary_id>
    get_drained_boundary_id_for_output() const override
    {
      return std::make_pair(2, 2);
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);
      (void)boundary_id;
      (void)direction;
      AssertThrow(
        false,
        ExcMessage(
          "Displacement loading not implemented for Ehlers verification examples."));

      return displ_incr;
    }
  };

  //@sect4{Derived class: Step load example}
  template <int dim>
  class VerificationEhlers1999StepLoad
    : public VerificationEhlers1999TubeBase<dim>
  {
  public:
    VerificationEhlers1999StepLoad(const Parameters::AllParameters &parameters)
      : VerificationEhlers1999TubeBase<dim>(parameters)
    {}

    virtual ~VerificationEhlers1999StepLoad()
    {}

  private:
    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          if (boundary_id == 2)
            {
              return this->parameters.load * N;
            }
        }

      (void)pt;

      return Tensor<1, dim>();
    }
  };

  //@sect4{Derived class: Load increasing example}
  template <int dim>
  class VerificationEhlers1999IncreaseLoad
    : public VerificationEhlers1999TubeBase<dim>
  {
  public:
    VerificationEhlers1999IncreaseLoad(
      const Parameters::AllParameters &parameters)
      : VerificationEhlers1999TubeBase<dim>(parameters)
    {}

    virtual ~VerificationEhlers1999IncreaseLoad()
    {}

  private:
    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          if (boundary_id == 2)
            {
              const double initial_load = this->parameters.load;
              const double final_load   = 20.0 * initial_load;
              const double initial_time = this->time->get_delta_t();
              const double final_time   = this->time->get_end();
              const double current_time = this->time->get_current();
              const double load =
                initial_load + (final_load - initial_load) *
                                 (current_time - initial_time) /
                                 (final_time - initial_time);
              return load * N;
            }
        }

      (void)pt;

      return Tensor<1, dim>();
    }
  };

  //@sect4{Class: Consolidation cube}
  template <int dim>
  class VerificationEhlers1999CubeConsolidation : public Solid<dim>
  {
  public:
    VerificationEhlers1999CubeConsolidation(
      const Parameters::AllParameters &parameters)
      : Solid<dim>(parameters)
    {}

    virtual ~VerificationEhlers1999CubeConsolidation()
    {}

  private:
    virtual void
    make_grid() override
    {
      GridGenerator::hyper_rectangle(this->triangulation,
                                     Point<dim>(0.0, 0.0, 0.0),
                                     Point<dim>(1.0, 1.0, 1.0),
                                     true);

      GridTools::scale(this->parameters.scale, this->triangulation);
      this->triangulation.refine_global(
        std::max(1U, this->parameters.global_refinement));

      typename Triangulation<dim>::active_cell_iterator
        cell = this->triangulation.begin_active(),
        endc = this->triangulation.end();
      for (; cell != endc; ++cell)
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            if (cell->face(face)->at_boundary() == true &&
                cell->face(face)->center()[2] == 1.0 * this->parameters.scale)
              {
                if (cell->face(face)->center()[0] <
                      0.5 * this->parameters.scale &&
                    cell->face(face)->center()[1] <
                      0.5 * this->parameters.scale)
                  cell->face(face)->set_boundary_id(100);
                else
                  cell->face(face)->set_boundary_id(101);
              }
        }
    }

    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.0 * this->parameters.scale;
      tracked_vertices[0][1] = 0.0 * this->parameters.scale;
      tracked_vertices[0][2] = 1.0 * this->parameters.scale;

      tracked_vertices[1][0] = 0.0 * this->parameters.scale;
      tracked_vertices[1][1] = 0.0 * this->parameters.scale;
      tracked_vertices[1][2] = 0.0 * this->parameters.scale;
    }

    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            101,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            101,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        0,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->x_displacement));

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        1,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->x_displacement));

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        2,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->y_displacement));

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        3,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->y_displacement));

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        4,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          if (boundary_id == 100)
            {
              return this->parameters.load * N;
            }
        }

      (void)pt;

      return Tensor<1, dim>();
    }

    virtual double
    get_prescribed_fluid_flow(const types::boundary_id &boundary_id,
                              const Point<dim>         &pt) const override
    {
      (void)pt;
      (void)boundary_id;
      return 0.0;
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 100;
    }

    virtual std::pair<types::boundary_id, types::boundary_id>
    get_drained_boundary_id_for_output() const override
    {
      return std::make_pair(101, 101);
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);
      (void)boundary_id;
      (void)direction;
      AssertThrow(
        false,
        ExcMessage(
          "Displacement loading not implemented for Ehlers verification examples."));

      return displ_incr;
    }
  };

  //@sect4{Franceschini experiments}
  template <int dim>
  class Franceschini2006Consolidation : public Solid<dim>
  {
  public:
    Franceschini2006Consolidation(const Parameters::AllParameters &parameters)
      : Solid<dim>(parameters)
    {}

    virtual ~Franceschini2006Consolidation()
    {}

  private:
    virtual void
    make_grid() override
    {
      const Point<dim - 1> mesh_center(0.0, 0.0);
      const double         radius = 0.5;
      // const double height = 0.27;  //8.1 mm for 30 mm radius
      const double           height = 0.23; // 6.9 mm for 30 mm radius
      Triangulation<dim - 1> triangulation_in;
      GridGenerator::hyper_ball(triangulation_in, mesh_center, radius);

      GridGenerator::extrude_triangulation(triangulation_in,
                                           2,
                                           height,
                                           this->triangulation);

      const CylindricalManifold<dim> cylinder_3d(2);
      const types::manifold_id       cylinder_id = 0;


      this->triangulation.set_manifold(cylinder_id, cylinder_3d);

      for (auto cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() == true)
                {
                  if (cell->face(face)->center()[2] == 0.0)
                    cell->face(face)->set_boundary_id(1);

                  else if (cell->face(face)->center()[2] == height)
                    cell->face(face)->set_boundary_id(2);

                  else
                    {
                      cell->face(face)->set_boundary_id(0);
                      cell->face(face)->set_all_manifold_ids(cylinder_id);
                    }
                }
            }
        }

      GridTools::scale(this->parameters.scale, this->triangulation);
      this->triangulation.refine_global(
        std::max(1U, this->parameters.global_refinement));
    }

    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.0 * this->parameters.scale;
      tracked_vertices[0][1] = 0.0 * this->parameters.scale;
      //  tracked_vertices[0][2] = 0.27*this->parameters.scale;
      tracked_vertices[0][2] = 0.23 * this->parameters.scale;

      tracked_vertices[1][0] = 0.0 * this->parameters.scale;
      tracked_vertices[1][1] = 0.0 * this->parameters.scale;
      tracked_vertices[1][2] = 0.0 * this->parameters.scale;
    }

    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            1,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            1,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        0,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement)));

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        1,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        2,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement)));
    }

    virtual double
    get_prescribed_fluid_flow(const types::boundary_id &boundary_id,
                              const Point<dim>         &pt) const override
    {
      (void)pt;
      (void)boundary_id;
      return 0.0;
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 2;
    }

    virtual std::pair<types::boundary_id, types::boundary_id>
    get_drained_boundary_id_for_output() const override
    {
      return std::make_pair(1, 2);
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);
      (void)boundary_id;
      (void)direction;
      AssertThrow(
        false,
        ExcMessage(
          "Displacement loading not implemented for Franceschini examples."));

      return displ_incr;
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          if (boundary_id == 2)
            {
              return (this->parameters.load * N);
              /*
              const double final_load = this->parameters.load;
              const double final_load_time = 10 * this->time->get_delta_t();
              const double current_time = this->time->get_current();


              const double c = final_load_time / 2.0;
              const double r = 200.0 * 0.03 / c;

              const double load = final_load * std::exp(r * current_time)
                                  / ( std::exp(c * current_time) +  std::exp(r *
              current_time)); return load * N;
              */
            }
        }

      (void)pt;

      return Tensor<1, dim>();
    }
  };

  // @sect3{Examples to reproduce experiments by Budday et al. 2017}
  // We group the definition of the geometry, boundary and loading conditions
  // specific to the examples to reproduce experiments by Budday et al. 2017
  // into specific classes.

  //@sect4{Base class: Cube geometry and loading pattern}
  template <int dim>
  class BrainBudday2017BaseCube : public Solid<dim>
  {
  public:
    BrainBudday2017BaseCube(const Parameters::AllParameters &parameters)
      : Solid<dim>(parameters)
    {}

    virtual ~BrainBudday2017BaseCube()
    {}

  private:
    virtual void
    make_grid() override
    {
      GridGenerator::hyper_cube(this->triangulation, 0.0, 1.0, true);

      typename Triangulation<dim>::active_cell_iterator
        cell = this->triangulation.begin_active(),
        endc = this->triangulation.end();
      for (; cell != endc; ++cell)
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            if (cell->face(face)->at_boundary() == true &&
                (cell->face(face)->boundary_id() == 0 ||
                 cell->face(face)->boundary_id() == 1 ||
                 cell->face(face)->boundary_id() == 2 ||
                 cell->face(face)->boundary_id() == 3))

              cell->face(face)->set_boundary_id(100);
        }

      GridTools::scale(this->parameters.scale, this->triangulation);
      this->triangulation.refine_global(
        std::max(1U, this->parameters.global_refinement));
    }

    virtual double
    get_prescribed_fluid_flow(const types::boundary_id &boundary_id,
                              const Point<dim>         &pt) const override
    {
      (void)pt;
      (void)boundary_id;
      return 0.0;
    }

    virtual std::pair<types::boundary_id, types::boundary_id>
    get_drained_boundary_id_for_output() const override
    {
      return std::make_pair(100, 100);
    }
  };

  //@sect4{Derived class: Uniaxial boundary conditions}
  template <int dim>
  class BrainBudday2017CubeTensionCompression
    : public BrainBudday2017BaseCube<dim>
  {
  public:
    BrainBudday2017CubeTensionCompression(
      const Parameters::AllParameters &parameters)
      : BrainBudday2017BaseCube<dim>(parameters)
    {}

    virtual ~BrainBudday2017CubeTensionCompression()
    {}

  private:
    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.5 * this->parameters.scale;
      tracked_vertices[0][1] = 0.5 * this->parameters.scale;
      tracked_vertices[0][2] = 1.0 * this->parameters.scale;

      tracked_vertices[1][0] = 0.5 * this->parameters.scale;
      tracked_vertices[1][1] = 0.5 * this->parameters.scale;
      tracked_vertices[1][2] = 0.5 * this->parameters.scale;
    }

    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            100,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            100,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        4,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->z_displacement));

      Point<dim> fix_node(0.5 * this->parameters.scale,
                          0.5 * this->parameters.scale,
                          0.0);
      typename DoFHandler<dim>::active_cell_iterator
        cell = this->dof_handler_ref.begin_active(),
        endc = this->dof_handler_ref.end();
      for (; cell != endc; ++cell)
        for (unsigned int node = 0; node < GeometryInfo<dim>::vertices_per_cell;
             ++node)
          {
            if ((abs(cell->vertex(node)[2] - fix_node[2]) <
                 (1e-6 * this->parameters.scale)) &&
                (abs(cell->vertex(node)[0] - fix_node[0]) <
                 (1e-6 * this->parameters.scale)))
              constraints.add_line(cell->vertex_dof_index(node, 0));

            if ((abs(cell->vertex(node)[2] - fix_node[2]) <
                 (1e-6 * this->parameters.scale)) &&
                (abs(cell->vertex(node)[1] - fix_node[1]) <
                 (1e-6 * this->parameters.scale)))
              constraints.add_line(cell->vertex_dof_index(node, 1));
          }

      if (this->parameters.load_type == "displacement")
        {
          const std::vector<double>  value = get_dirichlet_load(5, 2);
          FEValuesExtractors::Scalar direction;
          direction = this->z_displacement;

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            5,
            ConstantFunction<dim>(value[2], this->n_components),
            constraints,
            this->fe.component_mask(direction));
        }
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          if (boundary_id == 5)
            {
              const double final_load   = this->parameters.load;
              const double current_time = this->time->get_current();
              const double final_time   = this->time->get_end();
              const double num_cycles   = 3.0;

              return final_load / 2.0 *
                     (1.0 -
                      std::sin(
                        numbers::PI *
                        (2.0 * num_cycles * current_time / final_time + 0.5))) *
                     N;
            }
        }

      (void)pt;

      return Tensor<1, dim>();
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 5;
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 5) && (direction == 2))
        {
          const double final_displ    = this->parameters.load;
          const double current_time   = this->time->get_current();
          const double final_time     = this->time->get_end();
          const double delta_time     = this->time->get_delta_t();
          const double num_cycles     = 3.0;
          double       current_displ  = 0.0;
          double       previous_displ = 0.0;

          if (this->parameters.num_cycle_sets == 1)
            {
              current_displ =
                final_displ / 2.0 *
                (1.0 - std::sin(
                         numbers::PI *
                         (2.0 * num_cycles * current_time / final_time + 0.5)));
              previous_displ =
                final_displ / 2.0 *
                (1.0 - std::sin(numbers::PI *
                                (2.0 * num_cycles *
                                   (current_time - delta_time) / final_time +
                                 0.5)));
            }
          else
            {
              if (current_time <= (final_time * 1.0 / 3.0))
                {
                  current_displ =
                    final_displ / 2.0 *
                    (1.0 -
                     std::sin(numbers::PI * (2.0 * num_cycles * current_time /
                                               (final_time * 1.0 / 3.0) +
                                             0.5)));
                  previous_displ =
                    final_displ / 2.0 *
                    (1.0 -
                     std::sin(numbers::PI *
                              (2.0 * num_cycles * (current_time - delta_time) /
                                 (final_time * 1.0 / 3.0) +
                               0.5)));
                }
              else
                {
                  current_displ =
                    final_displ *
                    (1.0 -
                     std::sin(numbers::PI * (2.0 * num_cycles * current_time /
                                               (final_time * 2.0 / 3.0) -
                                             (num_cycles - 0.5))));
                  previous_displ =
                    final_displ *
                    (1.0 -
                     std::sin(numbers::PI *
                              (2.0 * num_cycles * (current_time - delta_time) /
                                 (final_time * 2.0 / 3.0) -
                               (num_cycles - 0.5))));
                }
            }
          displ_incr[2] = current_displ - previous_displ;
        }
      return displ_incr;
    }
  };

  //@sect4{Derived class: No lateral displacement in loading surfaces}
  template <int dim>
  class BrainBudday2017CubeTensionCompressionFullyFixed
    : public BrainBudday2017BaseCube<dim>
  {
  public:
    BrainBudday2017CubeTensionCompressionFullyFixed(
      const Parameters::AllParameters &parameters)
      : BrainBudday2017BaseCube<dim>(parameters)
    {}

    virtual ~BrainBudday2017CubeTensionCompressionFullyFixed()
    {}

  private:
    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.5 * this->parameters.scale;
      tracked_vertices[0][1] = 0.5 * this->parameters.scale;
      tracked_vertices[0][2] = 1.0 * this->parameters.scale;

      tracked_vertices[1][0] = 0.5 * this->parameters.scale;
      tracked_vertices[1][1] = 0.5 * this->parameters.scale;
      tracked_vertices[1][2] = 0.5 * this->parameters.scale;
    }

    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            100,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            100,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        4,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));


      if (this->parameters.load_type == "displacement")
        {
          const std::vector<double>  value = get_dirichlet_load(5, 2);
          FEValuesExtractors::Scalar direction;
          direction = this->z_displacement;

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            5,
            ConstantFunction<dim>(value[2], this->n_components),
            constraints,
            this->fe.component_mask(direction));

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            5,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement)));
        }
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          if (boundary_id == 5)
            {
              const double final_load   = this->parameters.load;
              const double current_time = this->time->get_current();
              const double final_time   = this->time->get_end();
              const double num_cycles   = 3.0;

              return final_load / 2.0 *
                     (1.0 -
                      std::sin(
                        numbers::PI *
                        (2.0 * num_cycles * current_time / final_time + 0.5))) *
                     N;
            }
        }

      (void)pt;

      return Tensor<1, dim>();
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 5;
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 5) && (direction == 2))
        {
          const double final_displ    = this->parameters.load;
          const double current_time   = this->time->get_current();
          const double final_time     = this->time->get_end();
          const double delta_time     = this->time->get_delta_t();
          const double num_cycles     = 3.0;
          double       current_displ  = 0.0;
          double       previous_displ = 0.0;

          if (this->parameters.num_cycle_sets == 1)
            {
              current_displ =
                final_displ / 2.0 *
                (1.0 - std::sin(
                         numbers::PI *
                         (2.0 * num_cycles * current_time / final_time + 0.5)));
              previous_displ =
                final_displ / 2.0 *
                (1.0 - std::sin(numbers::PI *
                                (2.0 * num_cycles *
                                   (current_time - delta_time) / final_time +
                                 0.5)));
            }
          else
            {
              if (current_time <= (final_time * 1.0 / 3.0))
                {
                  current_displ =
                    final_displ / 2.0 *
                    (1.0 -
                     std::sin(numbers::PI * (2.0 * num_cycles * current_time /
                                               (final_time * 1.0 / 3.0) +
                                             0.5)));
                  previous_displ =
                    final_displ / 2.0 *
                    (1.0 -
                     std::sin(numbers::PI *
                              (2.0 * num_cycles * (current_time - delta_time) /
                                 (final_time * 1.0 / 3.0) +
                               0.5)));
                }
              else
                {
                  current_displ =
                    final_displ *
                    (1.0 -
                     std::sin(numbers::PI * (2.0 * num_cycles * current_time /
                                               (final_time * 2.0 / 3.0) -
                                             (num_cycles - 0.5))));
                  previous_displ =
                    final_displ *
                    (1.0 -
                     std::sin(numbers::PI *
                              (2.0 * num_cycles * (current_time - delta_time) /
                                 (final_time * 2.0 / 3.0) -
                               (num_cycles - 0.5))));
                }
            }
          displ_incr[2] = current_displ - previous_displ;
        }
      return displ_incr;
    }
  };

  //@sect4{Derived class: No lateral or vertical displacement in loading
  // surface}
  template <int dim>
  class BrainBudday2017CubeShearFullyFixed : public BrainBudday2017BaseCube<dim>
  {
  public:
    BrainBudday2017CubeShearFullyFixed(
      const Parameters::AllParameters &parameters)
      : BrainBudday2017BaseCube<dim>(parameters)
    {}

    virtual ~BrainBudday2017CubeShearFullyFixed()
    {}

  private:
    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.75 * this->parameters.scale;
      tracked_vertices[0][1] = 0.5 * this->parameters.scale;
      tracked_vertices[0][2] = 0.0 * this->parameters.scale;

      tracked_vertices[1][0] = 0.25 * this->parameters.scale;
      tracked_vertices[1][1] = 0.5 * this->parameters.scale;
      tracked_vertices[1][2] = 0.0 * this->parameters.scale;
    }

    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            100,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            100,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        5,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));


      if (this->parameters.load_type == "displacement")
        {
          const std::vector<double>  value = get_dirichlet_load(4, 0);
          FEValuesExtractors::Scalar direction;
          direction = this->x_displacement;

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            4,
            ConstantFunction<dim>(value[0], this->n_components),
            constraints,
            this->fe.component_mask(direction));

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            4,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->y_displacement) |
             this->fe.component_mask(this->z_displacement)));
        }
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          if (boundary_id == 4)
            {
              const double           final_load   = this->parameters.load;
              const double           current_time = this->time->get_current();
              const double           final_time   = this->time->get_end();
              const double           num_cycles   = 3.0;
              const Point<3, double> axis(0.0, 1.0, 0.0);
              const double           angle = numbers::PI;
              static const Tensor<2, dim, double> R(
                Physics::Transformations::Rotations::rotation_matrix_3d(axis,
                                                                        angle));

              return (final_load *
                      (std::sin(2.0 * (numbers::PI)*num_cycles * current_time /
                                final_time)) *
                      (R * N));
            }
        }

      (void)pt;

      return Tensor<1, dim>();
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 4;
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 4) && (direction == 0))
        {
          const double final_displ    = this->parameters.load;
          const double current_time   = this->time->get_current();
          const double final_time     = this->time->get_end();
          const double delta_time     = this->time->get_delta_t();
          const double num_cycles     = 3.0;
          double       current_displ  = 0.0;
          double       previous_displ = 0.0;

          if (this->parameters.num_cycle_sets == 1)
            {
              current_displ =
                final_displ * (std::sin(2.0 * (numbers::PI)*num_cycles *
                                        current_time / final_time));
              previous_displ =
                final_displ *
                (std::sin(2.0 * (numbers::PI)*num_cycles *
                          (current_time - delta_time) / final_time));
            }
          else
            {
              AssertThrow(
                false,
                ExcMessage(
                  "Problem type not defined. Budday shear experiments implemented only for one set of cycles."));
            }
          displ_incr[0] = current_displ - previous_displ;
        }
      return displ_incr;
    }
  };

  //@sect4{Derived class: consolidation test}
  template <int dim>
  class BrainBudday2017CubeConsolidation : public BrainBudday2017BaseCube<dim>
  {
  public:
    BrainBudday2017CubeConsolidation(
      const Parameters::AllParameters &parameters)
      : BrainBudday2017BaseCube<dim>(parameters)
    {}

    virtual ~BrainBudday2017CubeConsolidation()
    {}

  private:
    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.5 * this->parameters.scale;
      tracked_vertices[0][1] = 0.5 * this->parameters.scale;
      tracked_vertices[0][2] = 1.0 * this->parameters.scale;

      tracked_vertices[1][0] = 0.5 * this->parameters.scale;
      tracked_vertices[1][1] = 0.5 * this->parameters.scale;
      tracked_vertices[1][2] = 0.5 * this->parameters.scale;
    }

    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            4,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            5,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            4,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            5,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        100,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement)));

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        4,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));
      if (this->parameters.load_type == "displacement")
        {
          const std::vector<double>  value = get_dirichlet_load(5, 2);
          FEValuesExtractors::Scalar direction;
          direction = this->z_displacement;

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            5,
            ConstantFunction<dim>(value[2], this->n_components),
            constraints,
            this->fe.component_mask(direction));

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            5,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement)));
        }
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          if (boundary_id == 5)
            {
              return (this->parameters.load * N);
            }
        }
      (void)pt;

      return Tensor<1, dim>();
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 5;
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);
      const double        current_time = this->time->get_current();
      const double        delta_time   = this->time->get_delta_t();

      if ((boundary_id == 5) && (direction == 2) &&
          (current_time <= delta_time))
        {
          displ_incr[2] = this->parameters.load;
        }
      return displ_incr;
    }
  };

  // @sect3{Examples to reproduce new rheometer experiments at LTM-FAU}
  // We group the definition of the geometry, boundary and loading conditions
  // specific to the examples to reproduce experiments at LTM-FAU into specific
  // classes.

  //@sect4{Base class: Sample geometry}
  template <int dim>
  class BrainRheometerLTMBase : public Solid<dim>
  {
  public:
    BrainRheometerLTMBase(const Parameters::AllParameters &parameters)
      : Solid<dim>(parameters)
    {}

    virtual ~BrainRheometerLTMBase()
    {}

  private:
    virtual void
    make_grid() override
    {
      const Point<dim - 1>   mesh_center(0.0, 0.0);
      const double           radius = this->parameters.radius;
      const double           height = this->parameters.height;
      Triangulation<dim - 1> triangulation_in;
      GridGenerator::hyper_ball(triangulation_in, mesh_center, radius);

      GridGenerator::extrude_triangulation(triangulation_in,
                                           2,
                                           height,
                                           this->triangulation);

      const CylindricalManifold<dim> cylinder_3d(2);
      const types::manifold_id       cylinder_id = 0;

      this->triangulation.set_manifold(cylinder_id, cylinder_3d);

      for (auto cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() == true)
                {
                  if (cell->face(face)->center()[2] == 0.0)
                    cell->face(face)->set_boundary_id(1);

                  else if (cell->face(face)->center()[2] == height)
                    cell->face(face)->set_boundary_id(2);

                  else
                    {
                      cell->face(face)->set_boundary_id(0);
                      cell->face(face)->set_all_manifold_ids(cylinder_id);
                    }
                }
            }
        }

      GridTools::scale(this->parameters.scale, this->triangulation);
      this->triangulation.refine_global(
        std::max(1U, this->parameters.global_refinement));
    }

    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 4.0 * this->parameters.scale;
      tracked_vertices[0][1] = 0.0 * this->parameters.scale;
      tracked_vertices[0][2] = 4.0 * this->parameters.scale;

      tracked_vertices[1][0] = 0.0 * this->parameters.scale;
      tracked_vertices[1][1] = 0.0 * this->parameters.scale;
      tracked_vertices[1][2] = 0.0 * this->parameters.scale;
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          AssertThrow(
            false,
            ExcMessage(
              "Pressure loading not implemented for rheometer examples."));
        }

      (void)boundary_id;
      (void)pt;
      (void)N;
      return Tensor<1, dim>();
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 2;
    }


    virtual double
    get_prescribed_fluid_flow(const types::boundary_id &boundary_id,
                              const Point<dim>         &pt) const override
    {
      (void)pt;
      (void)boundary_id;
      return 0.0;
    }

    virtual std::pair<types::boundary_id, types::boundary_id>
    get_drained_boundary_id_for_output() const override
    {
      if (this->parameters.lateral_drained == "drained" &&
          this->parameters.bottom_drained == "drained")
        {
          return std::make_pair(0, 1);
        }
      else if (this->parameters.lateral_drained == "drained")
        {
          return std::make_pair(0, 0);
        }
      else
        {
          return std::make_pair(1, 1);
        }
    }
  };



  //@sect4{Function to define Dirichlet boundary conditions for rotational
  // shear}
  template <int dim>
  class get_dirichlet_bc_LTMShear : public Function<dim>
  {
  public:
    get_dirichlet_bc_LTMShear(const double max_shear_load,
                              const double rotation_angle,
                              const double vertical_load)
      : Function<dim>(dim + 1)
      , max_shear_load(max_shear_load)
      , rotation_angle(rotation_angle)
      , vertical_load(vertical_load)
    {}

    void
    vector_value(const Point<dim> &pt, Vector<double> &values) const override
    {
      Assert(values.size() == (dim + 1),
             ExcDimensionMismatch(values.size(), (dim)));
      values = 0.0;

      const double pt_radius = std::sqrt(pt[0] * pt[0] + pt[1] * pt[1]);
      const double pt_theta  = std::atan2(pt[1], pt[0]) - rotation_angle;
      values(0)              = max_shear_load * pt_radius * std::sin(pt_theta);
      values(1) = -1.0 * max_shear_load * pt_radius * std::cos(pt_theta);
      values(2) = vertical_load;
    }

    void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>>   &value_list) const override
    {
      const unsigned int n_points = points.size();
      Assert(value_list.size() == n_points,
             ExcDimensionMismatch(value_list.size(), n_points));

      for (unsigned int p = 0; p < n_points; ++p)
        get_dirichlet_bc_LTMShear<dim>::vector_value(points[p], value_list[p]);
    }

  private:
    const double max_shear_load;
    const double rotation_angle;
    const double vertical_load;
  };

  //@sect4{Derived class: Cyclic rotational shear, only lateral boundaries are
  // drained}
  template <int dim>
  class BrainRheometerLTMShearLateralDrained : public BrainRheometerLTMBase<dim>
  {
  public:
    BrainRheometerLTMShearLateralDrained(
      const Parameters::AllParameters &parameters)
      : BrainRheometerLTMBase<dim>(parameters)
    {}

    virtual ~BrainRheometerLTMShearLateralDrained()
    {}

  private:
    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->pressure)));
        }

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        1,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));

      if (this->parameters.load_type == "displacement")
        {
          const std::vector<double> value = get_dirichlet_load(2, -1);
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            get_dirichlet_bc_LTMShear<dim>(value[0], value[1], value[2]),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement) |
             this->fe.component_mask(this->z_displacement)));
        }
    }


    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 2) && (direction == -1))
        {
          if (this->parameters.num_cycle_sets > 1)
            AssertThrow(
              false,
              ExcMessage(
                "Problem type not defined. Rheometer shear experiments implemented only for one set of cycles."));

          const double final_time   = this->time->get_end();
          const double delta_time   = this->time->get_delta_t();
          const double current_time = this->time->get_current();
          // const double total_angle   =
          // (this->parameters.load)*(numbers::PI)/180.; //parameter in grad
          const double total_angle =
            this->parameters.load; // parameter in rad, in this case phi=gamma
          const double num_cycles     = 3.0;
          double       current_angle  = 0.0;
          double       previous_angle = 0.0;

          // CHECK THAT dtheta < 0.5 degrees and small angle approx is valid
          current_angle =
            total_angle * (std::sin(2.0 * (numbers::PI)*num_cycles *
                                    current_time / final_time));
          previous_angle =
            total_angle * (std::sin(2.0 * (numbers::PI)*num_cycles *
                                    (current_time - delta_time) / final_time));

          // max magnitude of displ. increment in x-y plane (at max radius)
          displ_incr[0] = current_angle - previous_angle;
          // previous angle value needed to compute spatial distribution
          // of Dirichlet bcs correctly
          displ_incr[1] = previous_angle;
          // vertical load
          displ_incr[2] = 0;
        }
      return displ_incr;
    }
  };


  //@sect4{Derived class: Cyclic rotational shear, only lateral boundaries are
  // drained}
  template <int dim>
  class BrainRheometerLTMShearRelaxationLateralDrained
    : public BrainRheometerLTMBase<dim>
  {
  public:
    BrainRheometerLTMShearRelaxationLateralDrained(
      const Parameters::AllParameters &parameters)
      : BrainRheometerLTMBase<dim>(parameters)
    {}
    virtual ~BrainRheometerLTMShearRelaxationLateralDrained()
    {}

  private:
    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ZeroFunction<dim>(this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }

      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        1,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));

      if (this->parameters.load_type == "displacement")
        {
          const std::vector<double> value = get_dirichlet_load(2, -1);
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            get_dirichlet_bc_LTMShear<dim>(value[0], value[1], value[2]),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement) |
             this->fe.component_mask(this->z_displacement)));
        }
    }


    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 2) && (direction == -1))
        {
          if (this->parameters.num_cycle_sets > 1)
            AssertThrow(
              false,
              ExcMessage(
                "Problem type not defined. Rheometer shear experiments implemented only for one set of cycles."));

          // const double final_time    = this->time->get_end();
          const double delta_time      = this->time->get_delta_t();
          const double current_time    = this->time->get_current();
          const double final_load_time = this->parameters.end_load_time;
          // const double total_angle   =
          // (this->parameters.load)*(numbers::PI)/180.; //parameter in grad
          const double total_angle =
            this->parameters.load; // parameter in rad, in this case phi=gamma
          // const double num_cycles    = 3.0;
          double current_angle  = 0.0;
          double previous_angle = 0.0;

          if (current_time <= final_load_time)
            {
              // CHECK THAT dtheta < 0.5 degrees and small angle approx is valid
              current_angle = (current_time / final_load_time) * total_angle;
              if (current_time > delta_time)
                previous_angle =
                  ((current_time - delta_time) / final_load_time) * total_angle;
              // max magnitude of displ. increment in x-y plane (at max radius)
              displ_incr[0] = current_angle - previous_angle;
              // previous angle value needed to compute spatial distribution of
              // Dirichlet bcs correctly
              displ_incr[1] = previous_angle;
              // vertical load
              displ_incr[2] = 0;
            }
          else
            displ_incr = {0, 0, 0};
        }
      return displ_incr;
    }
  };


  //@sect4{Derived class: Cyclic Tension and Compression}
  template <int dim>
  class BrainRheometerLTMCyclicTensionCompression
    : public BrainRheometerLTMBase<dim>
  {
  public:
    BrainRheometerLTMCyclicTensionCompression(
      const Parameters::AllParameters &parameters)
      : BrainRheometerLTMBase<dim>(parameters)
    {}
    virtual ~BrainRheometerLTMCyclicTensionCompression()
    {}

  private:
    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      // Cylinder hull is drained
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ZeroFunction<dim>(this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }
      // Cylinder bottom is fully fixed in space (glued)
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        1,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));

      // Apply vertical displacement on cylinder top surface and fix in x- and
      // y-direction to account for glue
      if (this->parameters.load_type == "displacement")
        {
          const std::vector<double> value = get_dirichlet_load(2, 2);

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ConstantFunction<dim>(value[2], this->n_components),
            constraints,
            this->fe.component_mask(this->z_displacement));
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement)));
        }
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 2) && (direction == 2))
        {
          const double final_time   = this->time->get_end();
          const double delta_time   = this->time->get_delta_t();
          const double current_time = this->time->get_current();
          const double final_displ  = this->parameters.load;
          const double num_cycles   = this->parameters.num_cycle_sets;
          const double cycle_time   = final_time / (4 * num_cycles);
          const double displ_increment =
            (delta_time / cycle_time) * final_displ;

          if (current_time <= cycle_time)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 3 * cycle_time)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 5 * cycle_time)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 7 * cycle_time)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 9 * cycle_time)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 11 * cycle_time)
            displ_incr[2] = +displ_increment;
          else
            displ_incr[2] = -displ_increment;
        }
      return displ_incr;
    }
  };



  // Derived class: Cyclic Tension and Compression, read load data from file
  template <int dim>
  class BrainRheometerLTMCyclicTensionCompressionExp
    : public BrainRheometerLTMBase<dim>
  {
  public:
    BrainRheometerLTMCyclicTensionCompressionExp(
      const Parameters::AllParameters &parameters)
      : BrainRheometerLTMBase<dim>(parameters)
    {
      this->read_input_file(parameters.input_directory + '/' +
                            parameters.input_file);
    }
    virtual ~BrainRheometerLTMCyclicTensionCompressionExp()
    {}

    std::vector<double> displacement_data;

    void
    read_input_file(const std::string &filename)
    {
      using namespace dealii;

      efi::io::CSVReader<1> in(filename);

      in.read_header(efi::io::ignore_extra_column, "displacement");

      double displacement;
      while (in.read_row(displacement))
        {
          this->displacement_data.push_back(displacement);
        }
    }

  private:
    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      // Cylinder hull is drained
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ZeroFunction<dim>(this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }
      // Cylinder bottom is fully fixed in space (glued)
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        1,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));

      // Apply vertical displacement on cylinder top surface and fix in x- and
      // y-direction to account for glue
      if (this->parameters.load_type == "displacement")
        {
          const std::vector<double> value = get_dirichlet_load(2, 2);

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ConstantFunction<dim>(value[2], this->n_components),
            constraints,
            this->fe.component_mask(this->z_displacement));
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement)));
        }
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 2) && (direction == 2))
        {
          displ_incr[2] = -this->get_displacement(this->time->get_timestep());
        }
      return displ_incr;
    }

    double
    get_displacement(int timestep) const
    {
      return this->displacement_data[timestep] -
             displacement_data[timestep - 1];
    }
  };



  //@sect4{Derived class: Cyclic Tension and Compression}
  template <int dim>
  class BrainRheometerLTMCyclicTrapezoidalTensionCompression
    : public BrainRheometerLTMBase<dim>
  {
  public:
    BrainRheometerLTMCyclicTrapezoidalTensionCompression(
      const Parameters::AllParameters &parameters)
      : BrainRheometerLTMBase<dim>(parameters)
    {}
    virtual ~BrainRheometerLTMCyclicTrapezoidalTensionCompression()
    {}

  private:
    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      // Cylinder hull is drained
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ZeroFunction<dim>(this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }
      // Cylinder bottom is fully fixed in space (glued)
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        1,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));

      // Apply vertical displacement on cylinder top surface and fix in x- and
      // y-direction to account for glue
      if (this->parameters.load_type == "displacement")
        {
          const std::vector<double> value = get_dirichlet_load(2, 2);

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ConstantFunction<dim>(value[2], this->n_components),
            constraints,
            this->fe.component_mask(this->z_displacement));
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement)));
        }
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 2) && (direction == 2))
        {
          const double final_time   = this->time->get_end();
          const double delta_time   = this->time->get_delta_t();
          const double current_time = this->time->get_current();
          const double final_displ  = this->parameters.load;
          const double num_cycles   = this->parameters.num_cycle_sets;
          const double hold_time    = 7.00;
          const double cycle_time =
            (final_time - 2 * num_cycles * hold_time) / (4 * num_cycles); // 15s
          const double displ_increment =
            (delta_time / cycle_time) * final_displ; // 0.04mm/s

          if (current_time <= cycle_time)
            displ_incr[2] = -displ_increment;
          else if (current_time <= cycle_time + hold_time)
            displ_incr[2] = 0;
          else if (current_time <= 3 * cycle_time + hold_time)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 3 * cycle_time + 2 * hold_time)
            displ_incr[2] = 0;
          else if (current_time <= 5 * cycle_time + 2 * hold_time)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 5 * cycle_time + 3 * hold_time)
            displ_incr[2] = 0;
          else if (current_time <= 7 * cycle_time + 3 * hold_time)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 7 * cycle_time + 4 * hold_time)
            displ_incr[2] = 0;
          else if (current_time <= 9 * cycle_time + 4 * hold_time)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 9 * cycle_time + 5 * hold_time)
            displ_incr[2] = 0;
          else if (current_time <= 11 * cycle_time + 5 * hold_time)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 11 * cycle_time + 6 * hold_time)
            displ_incr[2] = 0;
          else
            displ_incr[2] = -displ_increment;
        }
      return displ_incr;
    }
  };


  //@sect4{Derived class: Relaxation}
  template <int dim>
  class BrainRheometerLTMRelaxationTensionCompression
    : public BrainRheometerLTMBase<dim>
  {
  public:
    BrainRheometerLTMRelaxationTensionCompression(
      const Parameters::AllParameters &parameters)
      : BrainRheometerLTMBase<dim>(parameters)
    {}
    virtual ~BrainRheometerLTMRelaxationTensionCompression()
    {}

  private:
    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      // Cylinder hull is drained
      if (this->parameters.lateral_drained == "drained")
        {
          if (this->time->get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                0,
                ConstantFunction<dim>(this->parameters.drained_pressure,
                                      this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
          else
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                0,
                ZeroFunction<dim>(this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
        }

      // Bottom drained
      if (this->parameters.bottom_drained == "drained")
        {
          if (this->time->get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                1,
                ConstantFunction<dim>(this->parameters.drained_pressure,
                                      this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
          else
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                1,
                ZeroFunction<dim>(this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
        }

      // Cylinder bottom is fully fixed in space (glued)
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        1,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));

      // Apply vertical displacement on cylinder top surface and fix in x- and
      // y-direction to account for glue
      if (this->parameters.load_type == "displacement")
        {
          const std::vector<double> value = get_dirichlet_load(2, 2);

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ConstantFunction<dim>(value[2], this->n_components),
            constraints,
            this->fe.component_mask(this->z_displacement));
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement)));
        }

      // Cylinder hull confined
      if (this->parameters.lateral_confined == "confined")
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement)));
        }
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(
        dim, 0.0); // vector of length dim with zero entries

      if ((boundary_id == 2) && (direction == 2))
        {
          const double final_displ     = this->parameters.load;
          const double final_load_time = this->parameters.end_load_time;
          const double current_time    = this->time->get_current();
          const double delta_time      = this->time->get_delta_t();

          double current_displ  = 0.0;
          double previous_displ = 0.0;

          if (current_time <= final_load_time)
            {
              current_displ = (current_time / final_load_time) * final_displ;

              if (current_time > delta_time)
                previous_displ =
                  ((current_time - delta_time) / final_load_time) * final_displ;

              displ_incr[2] = current_displ - previous_displ;
            }
          else
            displ_incr[2] = 0.0;
        }
      return displ_incr;
    }
  };


  //@sect4{Base class: Quarter Sample geometry}
  template <int dim>
  class BrainRheometerLTMBaseQuarter : public Solid<dim>
  {
  public:
    BrainRheometerLTMBaseQuarter(const Parameters::AllParameters &parameters)
      : Solid<dim>(parameters)
    {}
    virtual ~BrainRheometerLTMBaseQuarter()
    {}

  private:
    virtual void
    make_grid() override
    {
      const Point<dim - 1> mesh_center(0.0, 0.0);
      // const Point<dim> mesh_center2(0.0, 0.0, 0.0);
      const double radius = this->parameters.radius;
      const double height = this->parameters.height;

      // Create a quarter_hyper_ball in 2d, i.e. a quarter-circle and extrude it
      // to obtain a quarter cylinder
      Triangulation<dim - 1> triangulation_in;
      GridGenerator::quarter_hyper_ball(triangulation_in, mesh_center, radius);

      // Add a square to the quarter circle mesh
      /*Triangulation<dim-1> square;
      const std::vector<unsigned int> repetitions = {2, 2};
      const Point<dim-1> outer_edge(-4,4);
      GridGenerator::subdivided_hyper_rectangle(square, repetitions,
      mesh_center, outer_edge);

      Triangulation<dim-1> final_tria;
      GridGenerator::merge_triangulations(triangulation_in, square, final_tria,
      0.5, true);*/

      if (this->parameters.radius == 8)
        triangulation_in.refine_global(1);
      if (this->parameters.radius == 16)
        triangulation_in.refine_global(1);

      GridGenerator::extrude_triangulation(triangulation_in,
                                           3,
                                           height,
                                           this->triangulation);
      // GridGenerator::extrude_triangulation(final_tria, 3, height,
      // this->triangulation);

      // Assign a cylindrical manifold to the geometry
      const CylindricalManifold<dim> cylinder_3d(2);
      const types::manifold_id       cylinder_id = 0;
      // this->triangulation.reset_all_manifolds();
      // this->triangulation.set_all_manifold_ids_on_boundary(0,cylinder_id);
      this->triangulation.set_manifold(cylinder_id, cylinder_3d);


      // Assign proper boundary ids
      for (auto cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() == true)
                {
                  if (cell->face(face)->center()[2] == 0.0)
                    cell->face(face)->set_boundary_id(1); // bottom
                  else if (cell->face(face)->center()[2] == height)
                    cell->face(face)->set_boundary_id(2);        // top
                  else if (cell->face(face)->center()[0] == 0.0) //-4.0
                    cell->face(face)->set_boundary_id(3);        // left
                  else if (cell->face(face)->center()[1] == 0.0)
                    cell->face(face)->set_boundary_id(4); // front
                  else
                    {
                      cell->face(face)->set_boundary_id(0);
                      cell->face(face)->set_all_manifold_ids(cylinder_id);
                    }
                }
            }
        }

      GridTools::scale(this->parameters.scale, this->triangulation);
      this->triangulation.refine_global(
        std::max(1U, this->parameters.global_refinement));

      //				for (const auto &cell :
      // this->triangulation.active_cell_iterators()) { 					if
      // ((cell->center()[2] < 0.125*this->parameters.height ||
      // cell->center()[2] > 0.875*this->parameters.height) &&
      // std::sqrt((cell->center()[0])*(cell->center()[0]) +
      //(cell->center()[1])*(cell->center()[1])) > 0.8*this->parameters.radius)
      //						cell->set_refine_flag();
      //				}
      //				this->triangulation.execute_coarsening_and_refinement();
      //
      //				for (const auto &cell :
      // this->triangulation.active_cell_iterators()) { 					if
      // ((cell->center()[2] < 0.1*this->parameters.height || cell->center()[2]
      // > 0.9*this->parameters.height) &&
      // std::sqrt((cell->center()[0])*(cell->center()[0]) +
      //(cell->center()[1])*(cell->center()[1])) > 0.9*this->parameters.radius)
      //						cell->set_refine_flag();
      //				}
      //				this->triangulation.execute_coarsening_and_refinement();
      //
      //				for (const auto &cell :
      // this->triangulation.active_cell_iterators()) { 					if
      // ((cell->center()[2] < 0.05*this->parameters.height || cell->center()[2]
      // > 0.95*this->parameters.height) &&
      // std::sqrt((cell->center()[0])*(cell->center()[0]) +
      //(cell->center()[1])*(cell->center()[1])) > 0.95*this->parameters.radius)
      //						cell->set_refine_flag();
      //				}
      //				this->triangulation.execute_coarsening_and_refinement();


      // Assign proper boundary ids
      for (auto cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() == true)
                {
                  if (cell->face(face)->center()[2] == 0.0)
                    cell->face(face)->set_boundary_id(1); // bottom
                  else if (cell->face(face)->center()[2] == height)
                    cell->face(face)->set_boundary_id(2);         // top
                  else if (cell->face(face)->center()[0] < 1e-12) //-4.0
                    cell->face(face)->set_boundary_id(3);         // left
                  else if (cell->face(face)->center()[1] < 1e-12)
                    cell->face(face)->set_boundary_id(4); // front
                  else
                    {
                      cell->face(face)->set_boundary_id(0);
                      cell->face(face)->set_all_manifold_ids(cylinder_id);
                    }
                }
            }
        }
    }

    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.0 * this->parameters.scale;
      tracked_vertices[0][1] = 0.0 * this->parameters.scale;
      tracked_vertices[0][2] = this->parameters.height * this->parameters.scale;

      tracked_vertices[1][0] = this->parameters.radius * this->parameters.scale;
      tracked_vertices[1][1] = 0.0 * this->parameters.scale;
      tracked_vertices[1][2] =
        this->parameters.height / 2 * this->parameters.scale;
    }

    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      if (this->parameters.load_type == "displacement")
        {
          std::vector<bool> dof_touched(this->dof_handler_ref.n_dofs(), false);

          Quadrature<dim - 1> face_quadrature(
            this->fe.get_unit_face_support_points());
          FEFaceValues<dim> fe_values_face(this->mapping,
                                           this->fe,
                                           face_quadrature,
                                           update_quadrature_points);

          const unsigned int dofs_per_face   = this->fe.dofs_per_face;
          const unsigned int n_face_q_points = face_quadrature.size();

          std::vector<types::global_dof_index> dof_indices(dofs_per_face);

          for (const auto &cell : this->dof_handler_ref.active_cell_iterators())
            {
              if (!cell->is_artificial())
                {
                  for (const auto &face : cell->face_iterators())
                    {
                      if (face->at_boundary() && (face->boundary_id() == 1 ||
                                                  face->boundary_id() == 2))
                        {
                          fe_values_face.reinit(cell, face);
                          face->get_dof_indices(dof_indices);

                          for (unsigned int q_point = 0;
                               q_point < n_face_q_points;
                               ++q_point)
                            {
                              const unsigned int component =
                                this->fe.face_system_to_component_index(q_point)
                                  .first;
                              const unsigned int index_z = dof_indices[q_point];

                              if (component == 2 &&
                                  dof_touched[index_z] == false)
                                {
                                  dof_touched[index_z] = true;
                                  Point<dim> this_support_point =
                                    fe_values_face.quadrature_point(q_point);

                                  unsigned int index_x = 0;
                                  unsigned int index_y = 0;
                                  unsigned int index_p = 0;
                                  (void)index_p;

                                  for (unsigned int q_point = 0;
                                       q_point < n_face_q_points;
                                       ++q_point)
                                    {
                                      const Point<dim> this_support_point_2 =
                                        fe_values_face.quadrature_point(
                                          q_point);
                                      if (this_support_point ==
                                          this_support_point_2)
                                        {
                                          const unsigned int component =
                                            this->fe
                                              .face_system_to_component_index(
                                                q_point)
                                              .first;
                                          const unsigned int index =
                                            dof_indices[q_point];
                                          if (component == 0 &&
                                              dof_touched[index] == false)
                                            {
                                              dof_touched[index] = true;
                                              index_x            = index;
                                            }
                                          else if (component == 1 &&
                                                   dof_touched[index] == false)
                                            {
                                              dof_touched[index] = true;
                                              index_y            = index;
                                            }
                                          else if (component == 3 &&
                                                   dof_touched[index] == false)
                                            {
                                              dof_touched[index] = true;
                                              index_p            = index;
                                            }
                                        }
                                    }

                                  std::vector<double> displ_incr =
                                    get_dirichlet_load(2, 2);

                                  constraints.add_line(index_z);
                                  if (face->boundary_id() == 1)
                                    constraints.set_inhomogeneity(index_z, 0);
                                  if (face->boundary_id() == 2)
                                    constraints.set_inhomogeneity(
                                      index_z, displ_incr[2]);
                                  if (std::sqrt(this_support_point[0] *
                                                  this_support_point[0] +
                                                this_support_point[1] *
                                                  this_support_point[1]) <
                                      0.98 * this->parameters.radius)
                                    {
                                      constraints.add_line(index_x);
                                      constraints.set_inhomogeneity(index_x, 0);
                                      constraints.add_line(index_y);
                                      constraints.set_inhomogeneity(index_y, 0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

      // Cylinder hull is drained
      if (this->parameters.lateral_drained == "drained")
        {
          if (this->time->get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                0,
                ConstantFunction<dim>(this->parameters.drained_pressure,
                                      this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
          else
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                0,
                ZeroFunction<dim>(this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
        }

      // Cylinder hull is slowly undrained
      /*if (this->parameters.lateral_drained == "undrained") {
        if (this->time->get_timestep() < 2) {
            VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
              0,
            ZeroFunction<dim>(this->n_components),
              //ConstantFunction<dim>((this->time->get_timestep()-1)*10,this->n_components),
              constraints,
              this->fe.component_mask(this->pressure));
        }
      }*/

      // Bottom drained
      if (this->parameters.bottom_drained == "drained")
        {
          if (this->time->get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                1,
                ConstantFunction<dim>(this->parameters.drained_pressure,
                                      this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
          else
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                1,
                ZeroFunction<dim>(this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
        }

      // Top drained
      /*if (this->parameters.bottom_drained == "drained") {
        if (this->time->get_timestep() < 2) {
          VectorTools::interpolate_boundary_values(
              this->dof_handler_ref,
            2,
            ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        } else {
          VectorTools::interpolate_boundary_values(
              this->dof_handler_ref,
            2,
            ZeroFunction<dim>(this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }
      }*/

      if (this->parameters.load_type == "pressure")
        {
          // Cylinder bottom is fully fixed in space (glued)
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            1,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement) |
             this->fe.component_mask(this->z_displacement)));


          // Apply vertical displacement on cylinder top surface and fix in x-
          // and y-direction to account for glue
          if (this->parameters.load_type == "displacement")
            {
              const std::vector<double> value = get_dirichlet_load(2, 2);

              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                2,
                ConstantFunction<dim>(value[2], this->n_components),
                constraints,
                this->fe.component_mask(this->z_displacement));
            }
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement)));
        }


      // Define symmetry boundary conditions for lateral surfaces
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        3,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->x_displacement));
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        4,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->y_displacement));

      // Cylinder hull confined
      if (this->parameters.lateral_confined == "confined")
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement)));
        }
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        // AssertThrow(false, ExcMessage("Pressure loading not implemented for
        // rheometer examples."));

        (void)boundary_id;
      (void)pt;
      (void)N;
      return Tensor<1, dim>();
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 2;
    }

    virtual double
    get_prescribed_fluid_flow(const types::boundary_id &boundary_id,
                              const Point<dim>         &pt) const override
    {
      (void)pt;
      (void)boundary_id;
      return 0.0;
    }

    virtual std::pair<types::boundary_id, types::boundary_id>
    get_drained_boundary_id_for_output() const override
    {
      if (this->parameters.lateral_drained == "drained" &&
          this->parameters.bottom_drained == "drained")
        {
          return std::make_pair(0, 1);
        }
      else if (this->parameters.lateral_drained == "drained")
        {
          return std::make_pair(0, 0);
        }
      else
        {
          return std::make_pair(1, 1);
        }
    }

    // Define Dirichlet load, definition in derived classes
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override = 0;
  };


  //@sect4{Derived class: Cyclic Tension and Compression}
  template <int dim>
  class BrainRheometerLTMCyclicTensionCompressionQuarter
    : public BrainRheometerLTMBaseQuarter<dim>
  {
  public:
    BrainRheometerLTMCyclicTensionCompressionQuarter(
      const Parameters::AllParameters &parameters)
      : BrainRheometerLTMBaseQuarter<dim>(parameters)
    {}
    virtual ~BrainRheometerLTMCyclicTensionCompressionQuarter()
    {}

  private:
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 2) && (direction == 2))
        {
          const double final_time   = this->time->get_end();
          const double delta_time   = this->time->get_delta_t();
          const double current_time = this->time->get_current();
          const double final_displ  = this->parameters.load;
          const double num_cycles   = this->parameters.num_cycle_sets;
          const double cycle_time   = final_time / (4 * num_cycles);
          const double displ_increment =
            (delta_time / cycle_time) * final_displ;

          if (current_time <= cycle_time ||
              std::abs(current_time - cycle_time) < 1e-6)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 3 * cycle_time ||
                   std::abs(current_time - 3 * cycle_time) < 1e-6)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 5 * cycle_time ||
                   std::abs(current_time - 5 * cycle_time) < 1e-6)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 7 * cycle_time ||
                   std::abs(current_time - 7 * cycle_time) < 1e-6)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 9 * cycle_time ||
                   std::abs(current_time - 9 * cycle_time) < 1e-6)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 11 * cycle_time ||
                   std::abs(current_time - 11 * cycle_time) < 1e-6)
            displ_incr[2] = +displ_increment;
          else
            displ_incr[2] = -displ_increment;
        }
      return displ_incr;
    }
  };


  // Derived class: Cyclic Tension and Compression, read load data from file
  template <int dim>
  class BrainRheometerLTMCyclicTensionCompressionExpQuarter
    : public BrainRheometerLTMBaseQuarter<dim>
  {
  public:
    BrainRheometerLTMCyclicTensionCompressionExpQuarter(
      const Parameters::AllParameters &parameters)
      : BrainRheometerLTMBaseQuarter<dim>(parameters)
    {
      this->read_input_file(parameters.input_directory + '/' +
                            parameters.input_file);
    }
    virtual ~BrainRheometerLTMCyclicTensionCompressionExpQuarter()
    {}

    std::vector<double> displacement_data;

    void
    read_input_file(const std::string &filename)
    {
      using namespace dealii;

      efi::io::CSVReader<1> in(filename);

      in.read_header(efi::io::ignore_extra_column, "displacement");

      double displacement;
      while (in.read_row(displacement))
        {
          this->displacement_data.push_back(displacement);
        }
    }

  private:
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 2) && (direction == 2))
        {
          displ_incr[2] = this->get_displacement(this->time->get_timestep());
        }
      return displ_incr;
    }

    double
    get_displacement(int timestep) const
    {
      return this->displacement_data[timestep] -
             displacement_data[timestep - 1];
    }
  };


  //@sect4{Derived class: Cyclic Tension and Compression}
  template <int dim>
  class BrainRheometerLTMCyclicCompressionQuarter
    : public BrainRheometerLTMBaseQuarter<dim>
  {
  public:
    BrainRheometerLTMCyclicCompressionQuarter(
      const Parameters::AllParameters &parameters)
      : BrainRheometerLTMBaseQuarter<dim>(parameters)
    {}
    virtual ~BrainRheometerLTMCyclicCompressionQuarter()
    {}

  private:
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 2) && (direction == 2))
        {
          const double final_time   = this->time->get_end();
          const double delta_time   = this->time->get_delta_t();
          const double current_time = this->time->get_current();
          const double final_displ  = this->parameters.load;
          const double num_cycles   = this->parameters.num_cycle_sets;
          const double cycle_time   = final_time / (2 * num_cycles);
          const double displ_increment =
            (delta_time / cycle_time) * final_displ;

          if (current_time <= cycle_time)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 2 * cycle_time)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 3 * cycle_time)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 4 * cycle_time)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 5 * cycle_time)
            displ_incr[2] = -displ_increment;
          else
            displ_incr[2] = +displ_increment;
        }
      return displ_incr;
    }
  };


  //@sect4{Derived class: Cyclic Tension and Compression}
  template <int dim>
  class BrainRheometerLTMCyclicTensionQuarter
    : public BrainRheometerLTMBaseQuarter<dim>
  {
  public:
    BrainRheometerLTMCyclicTensionQuarter(
      const Parameters::AllParameters &parameters)
      : BrainRheometerLTMBaseQuarter<dim>(parameters)
    {}
    virtual ~BrainRheometerLTMCyclicTensionQuarter()
    {}

  private:
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 2) && (direction == 2))
        {
          const double final_time   = this->time->get_end();
          const double delta_time   = this->time->get_delta_t();
          const double current_time = this->time->get_current();
          const double final_displ  = this->parameters.load;
          const double num_cycles   = this->parameters.num_cycle_sets;
          const double cycle_time   = final_time / (2 * num_cycles);
          const double displ_increment =
            (delta_time / cycle_time) * final_displ;

          if (current_time <= cycle_time)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 2 * cycle_time)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 3 * cycle_time)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 4 * cycle_time)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 5 * cycle_time)
            displ_incr[2] = +displ_increment;
          else
            displ_incr[2] = -displ_increment;
        }
      return displ_incr;
    }
  };

  template <int dim>
  class BrainRheometerLTMCyclicTrapezoidalTensionCompressionQuarter
    : public BrainRheometerLTMBaseQuarter<dim>
  {
  public:
    BrainRheometerLTMCyclicTrapezoidalTensionCompressionQuarter(
      const Parameters::AllParameters &parameters)
      : BrainRheometerLTMBaseQuarter<dim>(parameters)
    {}
    virtual ~BrainRheometerLTMCyclicTrapezoidalTensionCompressionQuarter()
    {}

  private:
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 2) && (direction == 2))
        {
          const double final_time   = this->time->get_end();
          const double delta_time   = this->time->get_delta_t();
          const double current_time = this->time->get_current();
          const double final_displ  = this->parameters.load;
          const double num_cycles   = this->parameters.num_cycle_sets;
          const double hold_time    = 7.00;
          const double cycle_time =
            (final_time - 2 * num_cycles * hold_time) / (4 * num_cycles); // 15s
          const double displ_increment =
            (delta_time / cycle_time) * final_displ; // 0.04mm/s

          if (current_time <= cycle_time)
            displ_incr[2] = -displ_increment;
          else if (current_time <= cycle_time + hold_time)
            displ_incr[2] = 0;
          else if (current_time <= 3 * cycle_time + hold_time)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 3 * cycle_time + 2 * hold_time)
            displ_incr[2] = 0;
          else if (current_time <= 5 * cycle_time + 2 * hold_time)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 5 * cycle_time + 3 * hold_time)
            displ_incr[2] = 0;
          else if (current_time <= 7 * cycle_time + 3 * hold_time)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 7 * cycle_time + 4 * hold_time)
            displ_incr[2] = 0;
          else if (current_time <= 9 * cycle_time + 4 * hold_time)
            displ_incr[2] = -displ_increment;
          else if (current_time <= 9 * cycle_time + 5 * hold_time)
            displ_incr[2] = 0;
          else if (current_time <= 11 * cycle_time + 5 * hold_time)
            displ_incr[2] = +displ_increment;
          else if (current_time <= 11 * cycle_time + 6 * hold_time)
            displ_incr[2] = 0;
          else
            displ_incr[2] = -displ_increment;
        }
      return displ_incr;
    }
  };


  //@sect4{Derived class: Tension and Compression Relaxation}
  template <int dim>
  class BrainRheometerLTMRelaxationTensionCompressionQuarter
    : public BrainRheometerLTMBaseQuarter<dim>
  {
  public:
    BrainRheometerLTMRelaxationTensionCompressionQuarter(
      const Parameters::AllParameters &parameters)
      : BrainRheometerLTMBaseQuarter<dim>(parameters)
    {}
    virtual ~BrainRheometerLTMRelaxationTensionCompressionQuarter()
    {}

  private:
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(
        dim, 0.0); // vector of length dim with zero entries
      if (this->parameters.load_type == "displacement")
        {
          if ((boundary_id == 2) && (direction == 2))
            {
              const double final_displ     = this->parameters.load;
              const double final_load_time = this->parameters.end_load_time;
              const double current_time    = this->time->get_current();
              const double delta_time      = this->time->get_delta_t();

              double current_displ  = 0.0;
              double previous_displ = 0.0;

              if (current_time <= final_load_time)
                {
                  current_displ =
                    (current_time / final_load_time) * final_displ;

                  if (current_time > delta_time)
                    previous_displ =
                      ((current_time - delta_time) / final_load_time) *
                      final_displ;

                  displ_incr[2] = current_displ - previous_displ;
                }
              else
                displ_incr[2] = -0.000000001;

              /*//sinus load
              if (current_time <= final_load_time) {
                current_displ = 0.5*final_displ *
            std::sin((std::numbers::pi/final_load_time)*current_time-(final_load_time/4))+0.5*final_displ;

                if (current_time > delta_time)
                  previous_displ = 0.5*final_displ *
            std::sin((std::numbers::pi/final_load_time)*(current_time-delta_time)-(final_load_time/4))+0.5*final_displ;

                displ_incr[2] = current_displ - previous_displ;
              } else
                displ_incr[2] = 0.0;

              //rounded zig-zag load
              const double s = 0.99; //sharpness of edges from 0 to 1 (smooth to
            sharp) const double a = final_displ/(std::acos(-s)-std::acos(s))
            //amplitude if (current_time <= final_load_time) { current_displ = a
            * std::acos(s*(std::numbers::pi/final_load_time)*current_time)-a*std::acos(s);

              if (current_time > delta_time)
                previous_displ = a *
            std::acos(s*(std::numbers::pi/final_load_time)*(current_time-delta_time))-a*std::acos(s);

              displ_incr[2] = current_displ - previous_displ;
            } else
              displ_incr[2] = 0.0;*/
            }
        }
      return displ_incr;
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          if (boundary_id == 2)
            {
              const double final_load   = this->parameters.load;
              const double final_time   = this->parameters.end_load_time;
              const double current_time = this->time->get_current();
              double       load;

              // linear increasing load
              if (current_time <= final_time)
                {
                  load = final_load * (current_time / final_time);
                }
              else
                {
                  load = final_load;
                }

              // quadratic increasing load
              /*if (current_time <= final_time) {
                load = (final_load/(final_time*final_time)) *
              (current_time*current_time); } else { load = final_load;
              }*/

              return load * N;
            }
        }
      (void)pt;
      return Tensor<1, dim>();
    }
  };



  // @sect3{Examples to reproduce nanoindentation experiments for collaboration
  // with Pablo Sáez} We group the definition of the geometry, boundary and
  // loading conditions specific to the examples to reproduce nanoindentation
  // experiments into specific classes.

  //@sect4{Function to define Dirichlet boundary conditions for nanoindentation
  // using spherical indenter}
  template <int dim>
  class get_dirichlet_bc_BrainNanoSpherIndent : public Function<dim>
  {
  public:
    get_dirichlet_bc_BrainNanoSpherIndent(const double indent_dist_current,
                                          const double indent_dist_previous,
                                          const double indent_radius_max)
      : Function<dim>(dim + 1)
      , indent_dist_current(indent_dist_current)
      , // max indent dist (in % radius) for this dt
      indent_dist_previous(indent_dist_previous)
      , // max indent dist (in % radius) for previous dt
      indent_radius_max(indent_radius_max) // max indenter radius (in um)
    {}

    void
    vector_value(const Point<dim> &pt, Vector<double> &values) const override
    {
      Assert(values.size() == (dim + 1),
             ExcDimensionMismatch(values.size(), (dim)));

      values                 = 0.0;
      const double pt_radius = std::sqrt(pt[0] * pt[0] + pt[1] * pt[1]);
      const double indent_radius_current =
        std::sqrt(indent_radius_max * indent_radius_max -
                  ((1.0 - indent_dist_current) * indent_radius_max *
                   (1.0 - indent_dist_current) * indent_radius_max));

      const double indent_radius_previous =
        std::sqrt(indent_radius_max * indent_radius_max -
                  ((1.0 - indent_dist_previous) * indent_radius_max *
                   (1.0 - indent_dist_previous) * indent_radius_max));

      if (pt_radius <= indent_radius_previous)
        {
          values(2) = -1.0 * (indent_dist_current - indent_dist_previous) *
                      indent_radius_max;
        }
      else if (pt_radius > indent_radius_previous &&
               pt_radius <= indent_radius_current)
        {
          values(2) =
            -1.0 * (std::sqrt(indent_radius_max * indent_radius_max -
                              (pt_radius * pt_radius)) -
                    ((1.0 - indent_dist_current) * indent_radius_max));
        }
    }

    void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>>   &value_list) const override
    {
      const unsigned int n_points = points.size();
      Assert(value_list.size() == n_points,
             ExcDimensionMismatch(value_list.size(), n_points));

      for (unsigned int p = 0; p < n_points; ++p)
        get_dirichlet_bc_BrainNanoSpherIndent<dim>::vector_value(points[p],
                                                                 value_list[p]);
    }

  private:
    const double indent_dist_current;
    const double indent_dist_previous;
    const double indent_radius_max;
  };

  //@sect4{Base class: Sample geometry with spherical indenter}
  template <int dim>
  class BrainNanoSpherIndentBase : public Solid<dim>
  {
  public:
    BrainNanoSpherIndentBase(const Parameters::AllParameters &parameters)
      : Solid<dim>(parameters)
    {}

    virtual ~BrainNanoSpherIndentBase()
    {}

  private:
    virtual void
    make_grid() override
    {
      // Read external mesh
      GridIn<dim> gridin;
      gridin.attach_triangulation(this->triangulation);
      std::ifstream input_file("mesh.inp");
      gridin.read_abaqus(input_file);

      const double thickness       = 300;
      const double max_indent_dist = 0.1;

      const double indenter_radius = std::abs(this->parameters.load);
      const double max_indent_radius =
        (1.0 - max_indent_dist) * indenter_radius;


      // Assign boundary IDs
      for (auto cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() == true)
                {
                  if (cell->face(face)->center()[2] == 0.0) // bottom
                    cell->face(face)->set_boundary_id(0);

                  else if (cell->face(face)->center()[2] == thickness) // top
                    {
                      // Compute radius squared of the final loaded surface
                      // Based on indenter radius + 10% max indentation
                      // and adding a slight margin

                      const double margin = 1.0; // value above 1
                      const double max_bc_radius_sq =
                        indenter_radius * indenter_radius -
                        max_indent_radius * max_indent_radius;

                      if ((cell->face(face)->center()[0] *
                             cell->face(face)->center()[0] +
                           cell->face(face)->center()[1] *
                             cell->face(face)->center()[1]) <=
                          max_bc_radius_sq * margin)
                        cell->face(face)->set_boundary_id(
                          5); // final loaded surface
                      else
                        cell->face(face)->set_boundary_id(1);
                    }
                  else if (cell->face(face)->center()[1] == 0.0) // xz side
                    cell->face(face)->set_boundary_id(2);

                  else if (cell->face(face)->center()[0] == 0.0) // yz side
                    cell->face(face)->set_boundary_id(3);

                  else
                    cell->face(face)->set_boundary_id(
                      4); // rest of lateral sides
                }
            }
        }

      // Scale geometry
      GridTools::scale(this->parameters.scale, this->triangulation);

      // Refine mesh
      if (this->parameters.global_refinement > 0)
        this->triangulation.refine_global(this->parameters.global_refinement);
    }

    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.0 * this->parameters.scale;
      tracked_vertices[0][1] = 0.0 * this->parameters.scale;
      tracked_vertices[0][2] = 300. * this->parameters.scale; // thickness

      tracked_vertices[1][0] = 0.0 * this->parameters.scale;
      tracked_vertices[1][1] = 0.0 * this->parameters.scale;
      tracked_vertices[1][2] = 0.0 * this->parameters.scale;
    }

    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      // Top (unloaded) surface is drained
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            1,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            1,
            ZeroFunction<dim>(this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }

      // Define symmetry BCs for lateral surfaces
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        2,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->y_displacement));
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        3,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->x_displacement));
      // Fully fix bottom surface
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        0,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));

      if (this->parameters.load_type == "displacement")
        {
          if (this->time->get_current() <= this->parameters.end_load_time)
            {
              const std::vector<double> value = get_dirichlet_load(5, -1);
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                5,
                get_dirichlet_bc_BrainNanoSpherIndent<dim>(value[0],
                                                           value[1],
                                                           value[2]),
                constraints,
                this->fe.component_mask(this->z_displacement));
            }
          else
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                5,
                ZeroFunction<dim>(this->n_components),
                constraints,
                this->fe.component_mask(this->z_displacement));
            }
        }
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          AssertThrow(
            false,
            ExcMessage(
              "Pressure loading not implemented for nanometer examples."));
        }

      (void)boundary_id;
      (void)pt;
      (void)N;
      return Tensor<1, dim>();
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 5;
    }

    virtual double
    get_prescribed_fluid_flow(const types::boundary_id &boundary_id,
                              const Point<dim>         &pt) const override
    {
      (void)pt;
      (void)boundary_id;
      return 0.0;
    }

    virtual std::pair<types::boundary_id, types::boundary_id>
    get_drained_boundary_id_for_output() const override
    {
      return std::make_pair(1, 4);
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override = 0;
  };

  //@sect4{Derived class: One cycle of load and unload}
  template <int dim>
  class BrainNanoSpherIndentSinusoidalLoad
    : public BrainNanoSpherIndentBase<dim>
  {
  public:
    BrainNanoSpherIndentSinusoidalLoad(
      const Parameters::AllParameters &parameters)
      : BrainNanoSpherIndentBase<dim>(parameters)
    {}

    virtual ~BrainNanoSpherIndentSinusoidalLoad()
    {}

  private:
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 5) && (direction == -1)) // Nanoindentation
                                                   // experiments
        {
          if (this->parameters.num_cycle_sets > 1)
            AssertThrow(
              false,
              ExcMessage(
                "Problem type not defined. Nanoindentation experiments implemented only for one set of cycles."));

          const double indent_radius = std::abs(this->parameters.load);
          const double final_displ   = 0.1 * indent_radius;
          const double current_time  = this->time->get_current();
          const double final_time    = this->time->get_end();
          const double delta_time    = this->time->get_delta_t();
          const double num_cycles    = 1.0;

          double current_displ  = 0.0;
          double previous_displ = 0.0;

          current_displ =
            -1.0 * final_displ / 2.0 *
            (1.0 -
             std::sin(numbers::PI *
                      (2.0 * num_cycles * current_time / final_time + 0.5)));
          previous_displ =
            -1.0 * final_displ / 2.0 *
            (1.0 - std::sin(numbers::PI *
                            (2.0 * num_cycles * (current_time - delta_time) /
                               final_time +
                             0.5)));

          displ_incr[0] =
            current_displ - previous_displ; // max indent displ for this dt
          displ_incr[1] = indent_radius;    // indenter radius
          displ_incr[2] = 0.0;
        }

      return displ_incr;
    }
  };

  //@sect4{Derived class: Ramp load and maintain}
  template <int dim>
  class BrainNanoSpherIndentRampLoad : public BrainNanoSpherIndentBase<dim>
  {
  public:
    BrainNanoSpherIndentRampLoad(const Parameters::AllParameters &parameters)
      : BrainNanoSpherIndentBase<dim>(parameters)
    {}

    virtual ~BrainNanoSpherIndentRampLoad()
    {}

  private:
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 5) && (direction == -1)) // Nanoindentation
                                                   // experiments
        {
          const double indenter_radius = std::abs(this->parameters.load);
          const double max_indent_dist =
            0.1; // final max indentation is 10% of radius
          const double final_load_time = this->parameters.end_load_time;
          const double current_time    = this->time->get_current();
          const double delta_time      = this->time->get_delta_t();

          double current_displ  = 0.0;
          double previous_displ = 0.0;

          if (current_time <= final_load_time)
            {
              current_displ =
                (current_time / final_load_time) * max_indent_dist;

              if (current_time > delta_time)
                previous_displ =
                  ((current_time - delta_time) / final_load_time) *
                  max_indent_dist;
            }

          displ_incr[0] = current_displ; // max displ (% radius) in current time
          displ_incr[1] =
            previous_displ; // max displ (% radius) in previous time
          displ_incr[2] = indenter_radius; // indenter radius
        }

      return displ_incr;
    }
  };

  // Examples to reproduce nanoindentation experiments at LTM (FAU) with a flat
  // punch. We group the definition of the geometry, boundary and loading
  // conditions specific to the examples.

  // Base class: Sample geometry for flat punch
  template <int dim>
  class BrainNanoFlatPunchBase : public Solid<dim>
  {
  public:
    BrainNanoFlatPunchBase(const Parameters::AllParameters &parameters)
      : Solid<dim>(parameters)
    {}
    virtual ~BrainNanoFlatPunchBase()
    {}

  private:
    virtual void
    make_grid() override
    {
      // Use a subdivided_hyper_rectangle and make use of symmetry, unit [mm]
      // Define the diagonally opposite corner points, the origin of the
      // coordinate system is the front, lower left corner
      const double     width  = 4;
      const double     height = 2;
      const Point<dim> origin(0.0, 0.0, 0.0);
      const Point<dim> outside(width, width, height);
      // Define the indenter radius and the center of the applied displacement
      const double     indenter_radius = .9;
      const Point<dim> displ_center(0.0, 0.0, height);

      // Create a subdivided_hyper_rectangle with two cells along x- and
      // y-direction and one cell in z-direction
      const std::vector<unsigned int> repetitions = {2, 2, 1};
      GridGenerator::subdivided_hyper_rectangle(this->triangulation,
                                                repetitions,
                                                origin,
                                                outside);

      // Globally refine the mesh 2x to obtain 256 elements
      this->triangulation.refine_global(2);

      /*
      // obtain 3 cells inside the radius of 0.5cm on the top surface
      for (const auto &cell : this->triangulation.active_cell_iterators()) {
        for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
      { Point<dim> &v = cell->vertex(i); if (std::abs(v(0) - 0.5) < 1e-5) v(0)
      -= 0.332; else if (std::abs(v(0) - 1.0) < 1e-5) v(0) -= 0.666; else if
      (std::abs(v(0) - 1.5) < 1e-5) v(0) -= 0.98; else if (std::abs(v(0) - 2.0)
      < 1e-5) v(0) -= 1.17; else if (std::abs(v(0) - 2.5) < 1e-5) v(0) -= 1.02;
              else if (std::abs(v(0) - 3.0) < 1e-5)
                v(0) -= 0.7;
              else if (std::abs(v(0) - 3.5) < 1e-5)
                v(0) -= 0.4;
          }
      }

      for (const auto &cell : this->triangulation.active_cell_iterators()) {
        for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
      { Point<dim> &v = cell->vertex(i); if (std::abs(v(1) - 0.5) < 1e-5) v(1)
      -= 0.332; else if (std::abs(v(1) - 1.0) < 1e-5) v(1) -= 0.666; else if
      (std::abs(v(1) - 1.5) < 1e-5) v(1) -= 0.98; else if (std::abs(v(1) - 2.0)
      < 1e-5) v(1) -= 1.17; else if (std::abs(v(1) - 2.5) < 1e-5) v(1) -= 1.02;
              else if (std::abs(v(1) - 3.0) < 1e-5)
                v(1) -= 0.7;
              else if (std::abs(v(1) - 3.5) < 1e-5)
                v(1) -= 0.4;
          }
      }
      */

      // obtain 4 cells inside the radius of 9mm on the top surface -> move
      // vertices accordingly
      for (const auto &cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell;
               ++i)
            {
              Point<dim> &v = cell->vertex(i);
              if (std::abs(v(0) - .5) < 1e-5)
                v(0) -= .275;
              else if (std::abs(v(0) - 1.0) < 1e-5)
                v(0) -= .55;
              else if (std::abs(v(0) - 1.5) < 1e-5)
                v(0) -= .825;
              else if (std::abs(v(0) - 2.0) < 1e-5)
                v(0) -= 1.1;
              else if (std::abs(v(0) - 2.5) < 1e-5)
                v(0) -= 1.15;
              else if (std::abs(v(0) - 3.0) < 1e-5)
                v(0) -= .975;
              else if (std::abs(v(0) - 3.5) < 1e-5)
                v(0) -= .575;
            }
        }

      for (const auto &cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell;
               ++i)
            {
              Point<dim> &v = cell->vertex(i);
              if (std::abs(v(1) - .5) < 1e-5)
                v(1) -= .275;
              else if (std::abs(v(1) - 1.0) < 1e-5)
                v(1) -= .55;
              else if (std::abs(v(1) - 1.5) < 1e-5)
                v(1) -= .825;
              else if (std::abs(v(1) - 2.0) < 1e-5)
                v(1) -= 1.1;
              else if (std::abs(v(1) - 2.5) < 1e-5)
                v(1) -= 1.15;
              else if (std::abs(v(1) - 3.0) < 1e-5)
                v(1) -= .975;
              else if (std::abs(v(1) - 3.5) < 1e-5)
                v(1) -= .575;
            }
        }

      for (const auto &cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell;
               ++i)
            {
              Point<dim> &v = cell->vertex(i);
              if (std::abs(v(2) - .5) < 1e-5)
                v(2) += .375;
              else if (std::abs(v(2) - 1.0) < 1e-5)
                v(2) += .375;
              else if (std::abs(v(2) - 1.5) < 1e-5)
                v(2) += .25;
            }
        }

      // Globally refine (once to obtain 2048 elements)
      this->triangulation.refine_global(this->parameters.global_refinement);
      /*
      // Use external grid
      GridIn<dim> gridin;
      gridin.attach_triangulation(this->triangulation);
      std::ifstream input_file("mesh.inp");
      gridin.read_abaqus(input_file);

      const double width = 600;
      const double height = 300;
      const double indenter_radius = 0.2 * width;
      const Point<dim> displ_center(0.0, 0.0, height);
    */

      // Loop over all active cells to assign proper boundary ids
      for (auto cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() == true)
                {
                  if (cell->face(face)->center()[0] == 0.0)
                    cell->face(face)->set_boundary_id(0); // left
                  else if (cell->face(face)->center()[0] == width)
                    cell->face(face)->set_boundary_id(1); // right
                  else if (cell->face(face)->center()[1] == 0.0)
                    cell->face(face)->set_boundary_id(2); // front
                  else if (cell->face(face)->center()[1] == width)
                    cell->face(face)->set_boundary_id(1); // back
                  else if (cell->face(face)->center()[2] == 0.0)
                    cell->face(face)->set_boundary_id(4); // bottom
                  else if (cell->face(face)->center()[2] == height &&
                           displ_center.distance(cell->face(face)->center()) <
                             indenter_radius)
                    cell->face(face)->set_boundary_id(100); // loaded surface
                  else
                    {
                      cell->face(face)->set_boundary_id(5); // top
                    }
                }
            }
        }
    }

    // Define tracked vertices for post-processing
    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.0 * this->parameters.scale;
      tracked_vertices[0][1] = 0.0 * this->parameters.scale;
      tracked_vertices[0][2] =
        2.0 *
        this->parameters
          .scale; // TODO: use height instead of 3.5 or even top_center directly

      tracked_vertices[1][0] = 0.0 * this->parameters.scale;
      tracked_vertices[1][0] = 0.0 * this->parameters.scale;
      tracked_vertices[1][0] = 0.0 * this->parameters.scale;
    }

    // Apply Dirichlet constraints
    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      // Top (unloaded) surface is drained
      if (this->time->get_timestep() < 2)
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            5,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            1,
            ConstantFunction<dim>(this->parameters.drained_pressure,
                                  this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            5,
            ZeroFunction<dim>(this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            1,
            ZeroFunction<dim>(this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }

      // Fully fix bottom surface
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        4,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));

      // Define symmetry boundary conditions for lateral surfaces
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        0,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->x_displacement));
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        2,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->y_displacement));

      // Apply displacement to loaded top surface
      if (this->parameters.load_type == "displacement")
        {
          const std::vector<double> value = get_dirichlet_load(100, 2);

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            100,
            ConstantFunction<dim>(
              -value[2],
              this->n_components), // TODO: dont use minus sign, work with sign
                                   // function in get_dirichlet_load
            constraints,
            this->fe.component_mask(this->z_displacement));
        }
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          AssertThrow(
            false,
            ExcMessage(
              "Pressure loading not implemented for nanoindenter examples."));
        }

      (void)boundary_id;
      (void)pt;
      (void)N;
      return Tensor<1, dim>();
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 100;
    }

    virtual double
    get_prescribed_fluid_flow(const types::boundary_id &boundary_id,
                              const Point<dim>         &pt) const override
    {
      (void)pt;
      (void)boundary_id;
      return 0.0;
    }

    virtual std::pair<types::boundary_id, types::boundary_id>
    get_drained_boundary_id_for_output() const override
    {
      return std::make_pair(5, 5);
    }

    // Define Dirichlet load, definition in derived classes
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override = 0;
  };

  // Derived class to apply a ramp load and maintain
  template <int dim>
  class BrainNanoFlatPunchRampLoad : public BrainNanoFlatPunchBase<dim>
  {
  public:
    BrainNanoFlatPunchRampLoad(const Parameters::AllParameters &parameters)
      : BrainNanoFlatPunchBase<dim>(parameters)
    {}
    virtual ~BrainNanoFlatPunchRampLoad()
    {}

  private:
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(
        dim, 0.0); // vector of length dim with zero entries

      if ((boundary_id == 100) && (direction == 2))
        {
          const double final_displ = this->parameters.load;
          // const double final_load_time = this->parameters.end_load_time;
          const double final_load_time =
            10.0; // linear load increase from 0 to 10s
          const double start_deload_time = 20.0; // time to start deloading
          const double end_deload_time   = 30.0; // time to end deloading
          const double current_time      = this->time->get_current();
          const double delta_time        = this->time->get_delta_t();

          const double deload_time_steps =
            (end_deload_time - start_deload_time) / delta_time;

          double current_displ  = 0.0;
          double previous_displ = 0.0;

          if (current_time <= final_load_time)
            {
              current_displ = (current_time / final_load_time) * final_displ;

              if (current_time > delta_time)
                previous_displ =
                  ((current_time - delta_time) / final_load_time) * final_displ;

              displ_incr[2] = current_displ - previous_displ;
            }
          else if (current_time > start_deload_time &&
                   current_time <= end_deload_time)
            {
              displ_incr[2] = -final_displ / deload_time_steps;
            }
          else
            displ_incr[2] = 0.0;
        }

      return displ_incr;
    }
  };



  //@sect4{Base class: Quarter Sample geometry for Graz experiments}
  template <int dim>
  class HydroNanoGrazBaseQuarter : public Solid<dim>
  {
  public:
    HydroNanoGrazBaseQuarter(const Parameters::AllParameters &parameters)
      : Solid<dim>(parameters)
    {}
    virtual ~HydroNanoGrazBaseQuarter()
    {}

  private:
    virtual void
    make_grid() override
    {
      const Point<dim - 1> mesh_center(0.0, 0.0);
      const double         radius = this->parameters.radius; // 20.0;
      const double         height = this->parameters.height; // 20.0;

      // Define the indenter radius and the center of the applied displacement
      const double     indenter_radius = 4;
      const Point<dim> displ_center(0.0, 0.0, height);

      // Create a quarter_hyper_ball in 2d, i.e. a quarter-circle and extrude it
      // to obtain a quarter cylinder
      Triangulation<dim - 1> triangulation_in;
      GridGenerator::quarter_hyper_ball(triangulation_in, mesh_center, radius);
      // triangulation_in.refine_global(1);
      GridGenerator::extrude_triangulation(triangulation_in,
                                           3,
                                           height,
                                           this->triangulation);

      // Assign a cylindrical manifold to the geometry
      const CylindricalManifold<dim> cylinder_3d(
        2); // cylinder along axis=2 -> z-axis
      const types::manifold_id cylinder_id = 0;
      this->triangulation.set_manifold(cylinder_id, cylinder_3d);


      // Assign proper boundary ids
      for (auto cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() == true)
                {
                  if (cell->face(face)->center()[2] == 0.0)
                    cell->face(face)->set_boundary_id(1); // bottom
                  else if (cell->face(face)->center()[2] == height &&
                           displ_center.distance(cell->face(face)->center()) <
                             indenter_radius)
                    cell->face(face)->set_boundary_id(100); // loaded top
                  else if (cell->face(face)->center()[2] == height)
                    cell->face(face)->set_boundary_id(2);        // top
                  else if (cell->face(face)->center()[0] == 0.0) //-4.0
                    cell->face(face)->set_boundary_id(3);        // left
                  else if (cell->face(face)->center()[1] == 0.0)
                    cell->face(face)->set_boundary_id(4); // front
                  else
                    {
                      cell->face(face)->set_boundary_id(0);
                      cell->face(face)->set_all_manifold_ids(cylinder_id);
                    }
                }
            }
        }

      // Refine
      GridTools::scale(this->parameters.scale, this->triangulation);
      this->triangulation.refine_global(
        std::max(1U, this->parameters.global_refinement));


      /*for (const auto &cell : this->triangulation.active_cell_iterators()) {
        if (displ_center.distance(cell->center()) < 9)
          cell->set_refine_flag();
      }

      this->triangulation.execute_coarsening_and_refinement();

      for (const auto &cell : this->triangulation.active_cell_iterators()) {
        if (displ_center.distance(cell->center()) < 7)
          cell->set_refine_flag();
      }

      this->triangulation.execute_coarsening_and_refinement();

      for (const auto &cell : this->triangulation.active_cell_iterators()) {
        if (displ_center.distance(cell->center()) < 5 && cell->center()[2] >
      (height - 3)) cell->set_refine_flag();
      }

      this->triangulation.execute_coarsening_and_refinement();

      for (const auto &cell : this->triangulation.active_cell_iterators()) {
        if (displ_center.distance(cell->center()) < 2.8 && cell->center()[2] >
      (height - 2)) cell->set_refine_flag();
      }

      this->triangulation.execute_coarsening_and_refinement();

      for (const auto &cell : this->triangulation.active_cell_iterators()) {
        if (displ_center.distance(cell->center()) < 2.5 && cell->center()[2] >
      (height - 1)) cell->set_refine_flag();
      }

      this->triangulation.execute_coarsening_and_refinement();*/

      /*for (const auto &cell : this->triangulation.active_cell_iterators()) {
        if (displ_center.distance(cell->center()) < 2.4 &&
      displ_center.distance(cell->center()) > 1.6 && cell->center()[2] > (height
      - 0.7)) cell->set_refine_flag();
      }

      this->triangulation.execute_coarsening_and_refinement();

      for (const auto &cell : this->triangulation.active_cell_iterators()) {
        if (displ_center.distance(cell->center()) < 2.1 &&
      displ_center.distance(cell->center()) > 1.9 && cell->center()[2] > (height
      - 0.15)) cell->set_refine_flag();
      }

      this->triangulation.execute_coarsening_and_refinement();*/

      // Assign proper boundary ids
      for (auto cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() == true)
                {
                  if (cell->face(face)->center()[2] == 0.0)
                    cell->face(face)->set_boundary_id(1); // bottom
                  // else if (cell->face(face)->center()[2] == height &&
                  // displ_center.distance(cell->face(face)->center()) <
                  // indenter_radius*0.95)
                  //	cell->face(face)->set_boundary_id(100); //loaded top
                  else if (cell->face(face)->center()[2] == height &&
                           displ_center.distance(cell->face(face)->center()) <
                             indenter_radius)
                    cell->face(face)->set_boundary_id(
                      101); // loaded top boundary
                  else if (cell->face(face)->center()[2] == height)
                    cell->face(face)->set_boundary_id(2);        // top
                  else if (cell->face(face)->center()[0] == 0.0) //-4.0
                    cell->face(face)->set_boundary_id(3);        // left
                  else if (cell->face(face)->center()[1] == 0.0)
                    cell->face(face)->set_boundary_id(4); // front
                  else
                    {
                      cell->face(face)->set_boundary_id(0);
                      cell->face(face)->set_all_manifold_ids(cylinder_id);
                    }
                }
            }
        }
    }

    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.0 * this->parameters.scale;
      tracked_vertices[0][1] = 0.0 * this->parameters.scale;
      tracked_vertices[0][2] =
        this->parameters.height * this->parameters.scale; // height

      tracked_vertices[1][0] = 0.0 * this->parameters.scale;
      tracked_vertices[1][1] = 0.0 * this->parameters.scale;
      tracked_vertices[1][2] = 0.0 * this->parameters.scale;
    }

    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      // Cylinder hull is drained
      if (this->parameters.lateral_drained == "drained")
        {
          if (this->time->get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                0,
                ConstantFunction<dim>(this->parameters.drained_pressure,
                                      this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
          else
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                0,
                ZeroFunction<dim>(this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
        }

      // Bottom drained
      if (this->parameters.bottom_drained == "drained")
        {
          if (this->time->get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                1,
                ConstantFunction<dim>(this->parameters.drained_pressure,
                                      this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
          else
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                1,
                ZeroFunction<dim>(this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
        }

      // Unloaded Top drained
      if (this->parameters.bottom_drained == "drained")
        {
          if (this->time->get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                2,
                ConstantFunction<dim>(this->parameters.drained_pressure,
                                      this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
          else
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                2,
                ZeroFunction<dim>(this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
        }

      // Cylinder bottom is fully fixed in space (glued)
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        1,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));

      // Apply vertical displacement on cylinder top surface and fix in x- and
      // y-direction to account for glue
      if (this->parameters.load_type == "displacement")
        {
          /*const std::vector<double> value = get_dirichlet_load(100,2);

          VectorTools::interpolate_boundary_values(
              this->dof_handler_ref,
            100,
            ConstantFunction<dim>(value[2],this->n_components),
            constraints,
            this->fe.component_mask(this->z_displacement));
          VectorTools::interpolate_boundary_values(
              this->dof_handler_ref,
            100,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) | this->fe.component_mask(this->y_displacement)));*/

          const std::vector<double> value2 = get_dirichlet_load(101, 2);

          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            101,
            ConstantFunction<dim>(value2[2], this->n_components),
            constraints,
            this->fe.component_mask(this->z_displacement));
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            101,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement)));
        }

      // Define symmetry boundary conditions for lateral surfaces
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        3,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->x_displacement));
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        4,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->y_displacement));

      // Cylinder hull confined
      if (this->parameters.lateral_confined == "confined")
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement)));
        }
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        AssertThrow(
          false,
          ExcMessage(
            "Pressure loading not implemented for rheometer examples."));

      (void)boundary_id;
      (void)pt;
      (void)N;
      return Tensor<1, dim>();
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 101;
    }

    virtual double
    get_prescribed_fluid_flow(const types::boundary_id &boundary_id,
                              const Point<dim>         &pt) const override
    {
      (void)pt;
      (void)boundary_id;
      return 0.0;
    }

    virtual std::pair<types::boundary_id, types::boundary_id>
    get_drained_boundary_id_for_output() const override
    {
      if (this->parameters.lateral_drained == "drained" &&
          this->parameters.bottom_drained == "drained")
        {
          return std::make_pair(0, 1);
        }
      else if (this->parameters.lateral_drained == "drained")
        {
          return std::make_pair(0, 0);
        }
      else
        {
          return std::make_pair(1, 1);
        }
    }

    // Define Dirichlet load, definition in derived classes
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override = 0;
  };


  //@sect4{Derived class: Tension and Compression Relaxation}
  template <int dim>
  class HydroNanoGrazRelaxationCompressionQuarter
    : public HydroNanoGrazBaseQuarter<dim>
  {
  public:
    HydroNanoGrazRelaxationCompressionQuarter(
      const Parameters::AllParameters &parameters)
      : HydroNanoGrazBaseQuarter<dim>(parameters)
    {}
    virtual ~HydroNanoGrazRelaxationCompressionQuarter()
    {}

  private:
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(
        dim, 0.0); // vector of length dim with zero entries
      if (this->parameters.load_type == "displacement")
        {
          if ((boundary_id == 100 || boundary_id == 101) && (direction == 2))
            {
              const double final_displ     = this->parameters.load;
              const double final_load_time = this->parameters.end_load_time;
              const double current_time    = this->time->get_current();
              const double delta_time      = this->time->get_delta_t();

              double current_displ  = 0.0;
              double previous_displ = 0.0;

              if (current_time <= final_load_time)
                {
                  current_displ =
                    (current_time / final_load_time) * final_displ;

                  if (current_time > delta_time)
                    previous_displ =
                      ((current_time - delta_time) / final_load_time) *
                      final_displ;

                  displ_incr[2] = current_displ - previous_displ;
                }
              else
                displ_incr[2] = 0.0;
            }
        }
      return displ_incr;
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          if (boundary_id == 2)
            {
              const double final_load   = this->parameters.load;
              const double final_time   = this->parameters.end_load_time;
              const double current_time = this->time->get_current();
              double       load;

              if (current_time <= final_time)
                {
                  load = final_load * (current_time / final_time);
                }
              else
                {
                  load = final_load;
                }

              return load * N;
            }
        }
      (void)pt;
      return Tensor<1, dim>();
    }
  };

  template <int dim>
  class HydroNanoGrazRelaxationCompressionExpQuarter
    : public HydroNanoGrazBaseQuarter<dim>
  {
  public:
    HydroNanoGrazRelaxationCompressionExpQuarter(
      const Parameters::AllParameters &parameters)
      : HydroNanoGrazBaseQuarter<dim>(parameters)
    {
      this->read_input_file(parameters.input_directory + '/' +
                            parameters.input_file);
    }
    virtual ~HydroNanoGrazRelaxationCompressionExpQuarter()
    {}

    std::vector<double> displacement_data;

    void
    read_input_file(const std::string &filename)
    {
      using namespace dealii;

      efi::io::CSVReader<1> in(filename);

      in.read_header(efi::io::ignore_extra_column, "displacement");

      double displacement;
      while (in.read_row(displacement))
        {
          this->displacement_data.push_back(displacement);
        }
    }

  private:
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim, 0.0);

      if ((boundary_id == 100 || boundary_id == 101) && (direction == 2))
        {
          displ_incr[2] = this->get_displacement(this->time->get_timestep());
        }

      return displ_incr;
    }

    double
    get_displacement(int timestep) const
    {
      return this->displacement_data[timestep] -
             displacement_data[timestep - 1];
    }
  };


  //@sect4{Function to define Dirichlet boundary conditions for nanoindentation
  // using spherical indenter}
  template <int dim>
  class get_dirichlet_bc_HydroNanoGrazSpherIndent : public Function<dim>
  {
  public:
    get_dirichlet_bc_HydroNanoGrazSpherIndent(const double d_c,
                                              const double d_p,
                                              const double r_i)
      : Function<dim>(dim + 1)
      , d_c(d_c)
      , // current indentation depth
      d_p(d_p)
      ,        // previous indentation depth
      r_i(r_i) // indenter radius
    {}

    void
    vector_value(const Point<dim> &pt, Vector<double> &values) const
    {
      Assert(values.size() == (dim + 1),
             ExcDimensionMismatch(values.size(), (dim)));

      values = 0.0;
      // const double r_pt = std::sqrt(pt[0]*pt[0]+pt[1]*pt[1]);		// radius of
      // THIS point const double r_c =
      // std::sqrt(2*r_i*std::abs(d_c)-std::abs(d_c)*std::abs(d_c));		//
      // current indentation radius const double r_p =
      // std::sqrt(2*r_i*std::abs(d_p)-std::abs(d_p)*std::abs(d_p));		//
      // previous indentation radius const double dz_pt = 20 - pt[2]; 	//
      // current z-displacement of this point

      const double height = 20;
      const double z_0 =
        height + r_i + d_c; // current z-position of center of spherical
                            // indenter (d_c is negative)
      const double pt_z_c =
        z_0 -
        std::sqrt(
          (r_i * r_i) - (pt[0] * pt[0]) -
          (pt[1] *
           pt[1])); // current z-position of a point on the indenter surface

      if (pt[2] > pt_z_c)
        {
          values(2) = -1.0 * (pt[2] - pt_z_c);
          std::ofstream z_displ;
          z_displ.open("z_displ", std::ofstream::app);
          z_displ << std::setprecision(5) << std::scientific;
          z_displ << std::setw(16) << pt[0] << "," << std::setw(16) << pt[1]
                  << "," << std::setw(16) << pt[2] << "," << std::setw(16)
                  << z_0 << "," << std::setw(16) << pt_z_c << ","
                  << std::setw(16) << values(2) << std::endl;
          z_displ.close();
        }
      else
        std::cout << "point below indenter" << std::endl;

      /*double counter = 0; // count point that have boundary_id 100 but are
      outside the current indentation radius r_c and get 0 displ if (r_pt <=
      r_p) { values(2) = d_c - d_p; }  else if ( r_pt  > r_p && r_pt <= r_c) {
        values(2) = -1.0 * ((std::abs(d_c)-r_i +
      std::sqrt((r_i*r_i)-(r_pt*r_pt))) - dz_pt);

      } else {
        ++counter;
        std::cout << "Point " << pt[0] << " " << pt[1] << " " << pt[2] << "
      outside r_c!" << std::endl;
      }*/
    }

    void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>>   &value_list) const
    {
      const unsigned int n_points = points.size();
      Assert(value_list.size() == n_points,
             ExcDimensionMismatch(value_list.size(), n_points));

      for (unsigned int p = 0; p < n_points; ++p)
        get_dirichlet_bc_HydroNanoGrazSpherIndent<dim>::vector_value(
          points[p], value_list[p]);
    }

  private:
    const double d_c;
    const double d_p;
    const double r_i;
  };


  //@sect4{Base class: Quarter Sample geometry for Graz experiments}
  template <int dim>
  class HydroNanoGrazBaseQuarterSphere : public Solid<dim>
  {
  public:
    HydroNanoGrazBaseQuarterSphere(const Parameters::AllParameters &parameters)
      : Solid<dim>(parameters)
    {}
    virtual ~HydroNanoGrazBaseQuarterSphere()
    {}

  private:
    virtual void
    make_grid() override
    {
      const Point<dim - 1> mesh_center(0.0, 0.0);
      const double         radius = this->parameters.radius; // specimen radius
      const double         height = this->parameters.height; // specimen height
      const double         r_i    = 2;                       // indenter radius


      const double d_max = this->parameters.load; // maximal indentation depth

      // maximal indentation radius based on indenter radius and maximal
      // indentation depth
      const double r_max = std::sqrt(
        (2.0 * r_i * std::abs(d_max) - std::abs(d_max) * std::abs(d_max)));
      std::cout << "max indentation radius = " << r_max << std::endl;

      // center of the applied displacement
      const Point<dim> displ_center(0.0, 0.0, height);

      // Create a quarter_hyper_ball in 2d, i.e. a quarter-circle and extrude it
      // to obtain a quarter cylinder
      Triangulation<dim - 1> triangulation_in;
      GridGenerator::quarter_hyper_ball(triangulation_in, mesh_center, radius);
      GridGenerator::extrude_triangulation(triangulation_in,
                                           3,
                                           height,
                                           this->triangulation);

      // Assign a cylindrical manifold to the geometry
      const CylindricalManifold<dim> cylinder_3d(
        2); // cylinder along axis=2 -> z-axis
      const types::manifold_id cylinder_id = 0;
      this->triangulation.set_manifold(cylinder_id, cylinder_3d);


      int bottom = 0, top = 0, left = 0, front = 0, lateral = 0;
      for (auto cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() == true)
                {
                  if (cell->face(face)->center()[2] == 0.0)
                    {
                      cell->face(face)->set_boundary_id(1); // bottom
                      ++bottom;
                    }
                  else if (cell->face(face)->center()[2] == height)
                    {
                      cell->face(face)->set_boundary_id(2); // top surface
                      ++top;
                    }
                  else if (cell->face(face)->center()[0] == 0.0)
                    {
                      cell->face(face)->set_boundary_id(3); // left
                      ++left;
                    }
                  else if (cell->face(face)->center()[1] == 0.0)
                    {
                      cell->face(face)->set_boundary_id(4); // front
                      ++front;
                    }
                  else
                    {
                      cell->face(face)->set_boundary_id(
                        0); // remaining lateral faces
                      cell->face(face)->set_all_manifold_ids(cylinder_id);
                      ++lateral;
                    }
                  this->pcout << "id = " << cell->face(face)->boundary_id()
                              << ", x = " << cell->face(face)->center()[0]
                              << ", y = " << cell->face(face)->center()[1]
                              << ", z = " << cell->face(face)->center()[2]
                              << std::endl;
                }
            }
        }
      this->pcout << "bottom = " << bottom << ", top = " << top
                  << ", left = " << left << ", front = " << front
                  << ", lateral = " << lateral << std::endl;
      //				// Assign boundary IDs
      //				for (auto cell : this->triangulation.active_cell_iterators()) {
      //					for (unsigned int face = 0; face <
      // GeometryInfo<dim>::faces_per_cell; ++face) { 						if
      //(cell->face(face)->at_boundary() == true) { 							if
      //(cell->face(face)->center()[2] == 0.0) 	// bottom
      //								cell->face(face)->set_boundary_id(1);
      //							else if (cell->face(face)->center()[2] == height)
      //								cell->face(face)->set_boundary_id(2);	// top surface
      //							else if (cell->face(face)->center()[0] == 0.0)
      //								cell->face(face)->set_boundary_id(3); 	// left
      //							else if (cell->face(face)->center()[1] == 0.0)
      //								cell->face(face)->set_boundary_id(4); 	// front
      //							else {
      //								cell->face(face)->set_boundary_id(0);	// remaining
      // lateral faces
      // cell->face(face)->set_all_manifold_ids(cylinder_id);
      //							}
      //						}
      //					}
      //				}

      // Refine globally and locally
      GridTools::scale(this->parameters.scale, this->triangulation);
      this->triangulation.refine_global(
        std::max(1U, this->parameters.global_refinement));


      /*for (const auto &cell : this->triangulation.active_cell_iterators()) {
        if (displ_center.distance(cell->center()) < 9)
          cell->set_refine_flag();
      }

      this->triangulation.execute_coarsening_and_refinement();

      for (const auto &cell : this->triangulation.active_cell_iterators()) {
        if (displ_center.distance(cell->center()) < 7 && cell->center()[2] >
      (height - 4.5)) cell->set_refine_flag();
      }

      this->triangulation.execute_coarsening_and_refinement();

      for (const auto &cell : this->triangulation.active_cell_iterators()) {
        if (displ_center.distance(cell->center()) < 4 && cell->center()[2] >
      (height - 2.5)) cell->set_refine_flag();
      }

      this->triangulation.execute_coarsening_and_refinement();

      for (const auto &cell : this->triangulation.active_cell_iterators()) {
        if (displ_center.distance(cell->center()) < 2.8 && cell->center()[2] >
      (height - 1.2)) cell->set_refine_flag();
      }

      this->triangulation.execute_coarsening_and_refinement();*/

      /*for (const auto &cell : this->triangulation.active_cell_iterators()) {
        if (displ_center.distance(cell->center()) < 2 && cell->center()[2] >
      (height - 0.3)) cell->set_refine_flag();
      }

      this->triangulation.execute_coarsening_and_refinement();

      for (const auto &cell : this->triangulation.active_cell_iterators()) {
        if (displ_center.distance(cell->center()) < 1.9 && cell->center()[2] >
      (height - 0.1)) cell->set_refine_flag();
      }

      this->triangulation.execute_coarsening_and_refinement();*/


      // Assign boundary IDs
      bottom = 0, top = 0, left = 0, front = 0, lateral = 0;
      for (auto cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() == true)
                {
                  if (cell->face(face)->center()[2] == 0.0)
                    {
                      cell->face(face)->set_boundary_id(1); // bottom
                      ++bottom;
                    }
                  else if (cell->face(face)->center()[2] == height)
                    {
                      cell->face(face)->set_boundary_id(2); // top surface
                      ++top;
                    }
                  else if (cell->face(face)->center()[0] < 1e-12)
                    {
                      cell->face(face)->set_boundary_id(3); // left
                      ++left;
                    }
                  else if (cell->face(face)->center()[1] < 1e-12)
                    {
                      cell->face(face)->set_boundary_id(4); // front
                      ++front;
                    }
                  else
                    {
                      cell->face(face)->set_boundary_id(
                        0); // remaining lateral faces
                      cell->face(face)->set_all_manifold_ids(cylinder_id);
                      ++lateral;
                    }
                  this->pcout << "id = " << cell->face(face)->boundary_id()
                              << ", x = " << cell->face(face)->center()[0]
                              << ", y = " << cell->face(face)->center()[1]
                              << ", z = " << cell->face(face)->center()[2]
                              << std::endl;
                }
            }
        }
      this->pcout << "bottom = " << bottom << ", top = " << top
                  << ", left = " << left << ", front = " << front
                  << ", lateral = " << lateral << std::endl;
    }

    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.0 * this->parameters.scale;
      tracked_vertices[0][1] = 0.0 * this->parameters.scale;
      tracked_vertices[0][2] = this->parameters.height * this->parameters.scale;

      tracked_vertices[1][0] = 0.0 * this->parameters.scale;
      tracked_vertices[1][1] = 0.0 * this->parameters.scale;
      tracked_vertices[1][2] = 0.0 * this->parameters.scale;
    }

    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      if (true)
        { // Update boundary conditions
          // loaded surface cell counter
          /*const std::vector<double> r_c = get_dirichlet_load(100,2);

          std::cout << "current indenter radius = " << r_c[3] << std::endl;

          const double height = 20;	// specimen heigth
          const double r_i = 2;		// indenter radius
          const double z_0 = height + r_i + r_c[0]; 	// current z-position of
          indenter center (r_c[0]=d_c and is negative) const Point<dim>
          i_center(0.0,0.0,z_0); std::cout << "current indenter center z = " <<
          z_0 << std::endl;

          double cell_counter_add = 0;
          double cell_counter_remove = 0;
          for (auto cell : this->triangulation.active_cell_iterators()) {
            for (unsigned int face = 0; face <
          GeometryInfo<dim>::faces_per_cell; ++face) { if
          (cell->face(face)->at_boundary() == true) { if
          (cell->face(face)->boundary_id() == 2) { if
          (i_center.distance((cell->face(face)->center())) < r_i) // &&
          std::sqrt((cell->face(face)->center()[0]*cell->face(face)->center()[0]
          + cell->face(face)->center()[1]*cell->face(face)->center()[1])) <=
          r_c[3]-0.07)
                  {
                    cell->face(face)->set_boundary_id(100);
                    cell_counter_add++;		// add cells to loaded surface
                  }
                } else if (cell->face(face)->boundary_id() == 100) {
                  if (i_center.distance((cell->face(face)->center())) > r_i)
                  {
                    cell->face(face)->set_boundary_id(2);
                    cell_counter_remove++;		// remove cells from loaded
          surface
                  }
                }
              }
            }
          }
          std::cout << "number of cells with boundary ID 100 added = " <<
          cell_counter_add << std::endl; std::cout << "number of cells with
          boundary ID 100 removed = " << cell_counter_remove << std::endl;


          double cell_counter = 0;
          for (auto cell : this->triangulation.active_cell_iterators()) {
            for (unsigned int face = 0; face <
          GeometryInfo<dim>::faces_per_cell; ++face) { if
          (cell->face(face)->at_boundary() == true) { if
          (cell->face(face)->boundary_id() == 2) { if
          (std::sqrt((cell->face(face)->center()[0]*cell->face(face)->center()[0]
          + cell->face(face)->center()[1]*cell->face(face)->center()[1])) <=
          r_c[3]-0.07)
                  {
                    cell->face(face)->set_boundary_id(100);
                    cell_counter++;// final loaded surface
                  }
                }
              }
            }
          }
          std::cout << "number of cells with boundary ID 100 = " << cell_counter
          << std::endl;*/

          // Follow tutorial on plastic contact problem
          // A vector that stores the DoFs that are in touch
          std::vector<bool> dof_touched(this->dof_handler_ref.n_dofs(), false);
          // Create an instance of the solution vector
          this->distributed_solution = this->solution_n;
          // Clear the active set
          this->active_set.clear();

          // Loop over all DoFs to check whether they are in active contact or
          // not. Requires FEFaceValues and proper quadrature object. We create
          // this face quadrature object by choosing the "support points" of the
          // shape functions defined on the faces of cells. As a consequence, we
          // have as many quadrature points as there are shape functions per
          // face and looping over quadrature points is equivalent to looping
          // over shape functions defined on a face.
          Quadrature<dim - 1> face_quadrature(
            this->fe.get_unit_face_support_points());
          FEFaceValues<dim> fe_values_face(this->mapping,
                                           this->fe,
                                           face_quadrature,
                                           update_quadrature_points);

          const unsigned int dofs_per_face   = this->fe.dofs_per_face;
          const unsigned int n_face_q_points = face_quadrature.size();
          /*this->pcout << "Dofs per face: "<< dofs_per_face << " Quadrature
          points per face: " << n_face_q_points << std::endl; const unsigned int
          vertices_per_face = GeometryInfo<dim>::vertices_per_face; const
          unsigned int dofs_per_vertex = this->fe.dofs_per_vertex; const
          unsigned int lines_per_face = GeometryInfo<dim>::lines_per_face; const
          unsigned int dofs_per_line = this->fe.dofs_per_line; this->pcout <<
          "Vertices per face: "<< vertices_per_face << " Dofs per vertex: " <<
          dofs_per_vertex << std::endl; this->pcout << "Lines per face: "<<
          lines_per_face << " Dofs per line: " << dofs_per_line << std::endl;*/

          std::vector<types::global_dof_index> dof_indices(dofs_per_face);
          int                                  cnt = 0;

          for (const auto &cell : this->dof_handler_ref.active_cell_iterators())
            {
              if (!cell->is_artificial())
                {
                  for (const auto &face : cell->face_iterators())
                    {
                      if (face->at_boundary() && face->boundary_id() == 2)
                        {
                          fe_values_face.reinit(cell, face);
                          face->get_dof_indices(dof_indices);

                          // At each quadrature point (i.e., at each support
                          // point of a degree of freedom located on the contact
                          // boundary), we then ask whether it is part of the
                          // z-displacement degrees of freedom and if we haven't
                          // encountered this degree of freedom yet (which can
                          // happen for those on the edges between faces), we
                          // need to evaluate the gap between the deformed
                          // object and the obstacle. If the active set
                          // condition is true, then we add a constraint to the
                          // AffineConstraints object that the next Newton
                          // update needs to satisfy, set the solution vector's
                          // corresponding element to the correct value, and add
                          // the index to the IndexSet object that stores which
                          // degree of freedom is part of the contact:

                          for (unsigned int q_point = 0;
                               q_point < n_face_q_points;
                               ++q_point)
                            {
                              const unsigned int component =
                                this->fe.face_system_to_component_index(q_point)
                                  .first;
                              const unsigned int index_z = dof_indices[q_point];
                              // this->pcout << "Index: "<< index_z << "
                              // Component: " << component << std::endl;

                              if (component == 2 &&
                                  dof_touched[index_z] == false)
                                {
                                  dof_touched[index_z] = true;
                                  Point<dim> this_support_point =
                                    fe_values_face.quadrature_point(q_point);

                                  // displacement vector of this support point
                                  Tensor<1, dim> solution_here;
                                  unsigned int   index_x = 0;
                                  unsigned int   index_y = 0;
                                  unsigned int   index_p = 0;
                                  (void)index_p;

                                  for (unsigned int q_point = 0;
                                       q_point < n_face_q_points;
                                       ++q_point)
                                    {
                                      const Point<dim> this_support_point_2 =
                                        fe_values_face.quadrature_point(
                                          q_point);
                                      if (this_support_point ==
                                          this_support_point_2)
                                        {
                                          const unsigned int component =
                                            this->fe
                                              .face_system_to_component_index(
                                                q_point)
                                              .first;
                                          const unsigned int index =
                                            dof_indices[q_point];
                                          if (component == 0 &&
                                              dof_touched[index] == false)
                                            {
                                              dof_touched[index] = true;
                                              solution_here[0] =
                                                this->solution_n(index);
                                              index_x = index;
                                            }
                                          else if (component == 1 &&
                                                   dof_touched[index] == false)
                                            {
                                              dof_touched[index] = true;
                                              solution_here[1] =
                                                this->solution_n(index);
                                              index_y = index;
                                            }
                                          else if (component == 3 &&
                                                   dof_touched[index] == false)
                                            {
                                              dof_touched[index] = true;
                                              index_p            = index;
                                            }
                                        }
                                    }

                                  // const double obstacle_value =
                                  // obstacle->value(this_support_point, 2);  //
                                  // TODO
                                  std::vector<double> obstacle_value =
                                    get_dirichlet_load(100, 2);
                                  solution_here[2] = this->solution_n(
                                    index_z); // TODO Block vector???
                                  // this_support_point = this_support_point +
                                  // solution_here; //remove comment this->pcout
                                  // << "x = " << this_support_point[0] << ", y
                                  // = " << this_support_point[1] << ", z = " <<
                                  // this_support_point[2] << std::endl;
                                  // this->pcout << "idx = " << index_x << ",
                                  // idy = " << index_y << ", idz = " << index_z
                                  // << std::endl;

                                  /*const double z_0 = this->parameters.height +
              obstacle_value[2] + obstacle_value[0]; 	// current z-position of
              center of spherical indenter (d_c is negative) const Point<dim>
              indent_center(0,0,z_0); const double pt_z_c = z_0 -
              std::sqrt((obstacle_value[2]*obstacle_value[2]) -
              (this_support_point[0]*this_support_point[0]) -
              (this_support_point[1]*this_support_point[1])); 	// current
              z-position of a point on the indenter surface

                                  if (indent_center.distance(this_support_point)
              <= obstacle_value[2] && this_support_point[2] >= pt_z_c) { const
              double undeformed_gap = pt_z_c-this_support_point[2];
                                    constraints.add_line(index_z);
                                    constraints.set_inhomogeneity(index_z,
              undeformed_gap);
              //											constraints.add_line(index_x);
              //											constraints.set_inhomogeneity(index_x, 0);
              //											constraints.add_line(index_y);
              //											constraints.set_inhomogeneity(index_y, 0);
                                    std::ofstream z_displ;
                                    z_displ.open(this->parameters.output_directory
              + "/z_displ", std::ofstream::app); z_displ << std::setprecision(6)
              << std::scientific; z_displ << std::setw(16) <<
              this->time->get_current() << ","
                                        << std::setw(16) << index_z << ","
                                        << std::setw(16) << undeformed_gap <<
              std::endl; z_displ.close(); this->distributed_solution(index_z) =
              undeformed_gap; this->active_set.add_index(index_z); } else {
                                    constraints.add_line(index_p);
                                    constraints.set_inhomogeneity(index_p, 0);
                                  }*/


                                  /*// flat-punch
                                  const double flat_punch_height =
                                  this->parameters.height + obstacle_value[0];
                                  const Point<dim>
                                  flat_punch_center(0,0,flat_punch_height); if
                                  (flat_punch_center.distance(this_support_point)
                                  <= obstacle_value[2]) { const double
                                  undeformed_gap =
                                  flat_punch_height-this_support_point[2];
                                    constraints.add_line(index_z);
                                    constraints.set_inhomogeneity(index_z,
                                  undeformed_gap);
                                    //constraints.add_line(index_x);
                                    //constraints.set_inhomogeneity(index_x, 0);
                                    //constraints.add_line(index_y);
                                    //constraints.set_inhomogeneity(index_y, 0);
                                    std::ofstream z_displ;
                                    z_displ.open(this->parameters.output_directory
                                  + "/z_displ", std::ofstream::app); z_displ <<
                                  std::setprecision(6) << std::scientific;
                                    z_displ << std::setw(16) <<
                                  this->time->get_current() << ","
                                        << std::setw(16) << index_z << ","
                                        << std::setw(16) << undeformed_gap <<
                                  std::endl; z_displ.close();
                                    this->distributed_solution(index_z) =
                                  undeformed_gap;
                                    this->active_set.add_index(index_z);
                                  }*/

                                  // whole surface
                                  // const double undeformed_gap =
                                  // this->parameters.height + obstacle_value[0]
                                  // - this_support_point[2];
                                  const double undeformed_gap =
                                    obstacle_value[0] - obstacle_value[1];
                                  constraints.add_line(index_z);
                                  constraints.set_inhomogeneity(index_z,
                                                                undeformed_gap);
                                  if (std::sqrt(this_support_point[0] *
                                                  this_support_point[0] +
                                                this_support_point[1] *
                                                  this_support_point[1]) <
                                      0.98 * this->parameters.radius)
                                    {
                                      constraints.add_line(index_x);
                                      constraints.set_inhomogeneity(index_x, 0);
                                      constraints.add_line(index_y);
                                      constraints.set_inhomogeneity(index_y, 0);
                                      ++cnt;
                                    }
                                  std::ofstream z_displ;
                                  z_displ.open(
                                    this->parameters.output_directory +
                                      "/z_displ",
                                    std::ofstream::app);
                                  z_displ << std::setprecision(6)
                                          << std::scientific;
                                  z_displ << std::setw(16)
                                          << this->time->get_current() << ","
                                          << std::setw(16) << index_z << ","
                                          << std::setw(16) << undeformed_gap
                                          << std::endl;
                                  z_displ.close();
                                  this->distributed_solution(index_z) =
                                    undeformed_gap;
                                  this->active_set.add_index(index_z);



                                  // point radius from center axis
                                  /*const double this_support_point_radius =
                                  std::sqrt((this_support_point[0]*this_support_point[0])
                                  +
                                  (this_support_point[1]*this_support_point[1]));
                                  //check if point is under indenter if
                                  (this_support_point_radius <=
                                  obstacle_value[2]) {
                                    //const double undeformed_gap =
                                  obstacle_value - this_support_point(2); const
                                  double undeformed_gap =
                                  obstacle_value[0]-obstacle_value[1];
                                    //this->pcout << this_support_point(2) <<
                                  std::endl;

                                    constraints.add_line(index_z);
                                    constraints.set_inhomogeneity(index_z,
                                  undeformed_gap);
                                    this->distributed_solution(index_z) =
                                  undeformed_gap;
                                    this->active_set.add_index(index_z);
                                  }*/
                                }
                            }
                        }
                    }
                }
            }
          this->pcout << "counter: " << cnt << std::endl;
          this->distributed_solution.compress(VectorOperation::insert);
          // this->solution_n = this->distributed_solution;

          this->pcout
            << "Size of active set: "
            << Utilities::MPI::sum(
                 (this->active_set & this->locally_owned_dofs).n_elements(),
                 this->mpi_communicator)
            << std::endl;
        }

      // Cylinder hull is drained
      if (this->parameters.lateral_drained == "drained")
        {
          if (this->time->get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                0,
                ConstantFunction<dim>(this->parameters.drained_pressure,
                                      this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
          else
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                0,
                ZeroFunction<dim>(this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
        }

      // Bottom drained
      if (this->parameters.bottom_drained == "drained")
        {
          if (this->time->get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                1,
                ConstantFunction<dim>(this->parameters.drained_pressure,
                                      this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
          else
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                1,
                ZeroFunction<dim>(this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
        }

      /*// Unloaded Top drained
      if (this->parameters.bottom_drained == "drained") {
        if (this->time->get_timestep() < 2) {
          VectorTools::interpolate_boundary_values(
              this->dof_handler_ref,
              2,
              ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
              constraints,
              this->fe.component_mask(this->pressure));
        } else {
          VectorTools::interpolate_boundary_values(
              this->dof_handler_ref,
              2,
              ZeroFunction<dim>(this->n_components),
              constraints,
              this->fe.component_mask(this->pressure));
        }
      }*/

      // Define symmetry boundary conditions for lateral surfaces
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        3,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->x_displacement));
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        4,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->y_displacement));

      // Cylinder hull confined
      if (this->parameters.lateral_confined == "confined")
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement)));
        }

      // Cylinder bottom is fully fixed in space (glued)
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        1,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));

      /*// indenter tip
    if (this->parameters.load_type == "displacement") {
      //if (this->time->get_current()<=this->parameters.end_load_time) {
        const std::vector<double> value = get_dirichlet_load(100,2);
        const double displ = value[0] - value[1];

        VectorTools::interpolate_boundary_values(
             this->dof_handler_ref,
             2,
             ConstantFunction<dim>(displ,this->n_components),
             constraints,
             this->fe.component_mask(this->z_displacement));

        VectorTools::interpolate_boundary_values(
             this->dof_handler_ref,
             2,
             ZeroFunction<dim>(this->n_components),
             constraints,
             (this->fe.component_mask(this->x_displacement) | this->fe.component_mask(this->y_displacement)));
      //}
    }*/
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          AssertThrow(
            false,
            ExcMessage(
              "Pressure loading not implemented for nanometer examples."));
        }

      (void)boundary_id;
      (void)pt;
      (void)N;
      return Tensor<1, dim>();
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 2;
    }

    virtual double
    get_prescribed_fluid_flow(const types::boundary_id &boundary_id,
                              const Point<dim>         &pt) const override
    {
      (void)pt;
      (void)boundary_id;
      return 0.0;
    }

    virtual std::pair<types::boundary_id, types::boundary_id>
    get_drained_boundary_id_for_output() const override
    {
      return std::make_pair(1, 2);
    }

    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override = 0;
  };



  //@sect4{Derived class: Tension and Compression Relaxation}
  template <int dim>
  class HydroNanoGrazRelaxationCompressionQuarterSphere
    : public HydroNanoGrazBaseQuarterSphere<dim>
  {
  public:
    HydroNanoGrazRelaxationCompressionQuarterSphere(
      const Parameters::AllParameters &parameters)
      : HydroNanoGrazBaseQuarterSphere<dim>(parameters)
    {}
    virtual ~HydroNanoGrazRelaxationCompressionQuarterSphere()
    {}

  private:
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(dim + 1, 0.0);

      if ((boundary_id == 100) && (direction == 2))
        {
          const double r_i = 2; // indenter radius
          const double d_max =
            this->parameters.load; // maximal indetation depth
          const double final_load_time = this->parameters.end_load_time;
          const double current_time    = this->time->get_current();
          const double delta_time      = this->time->get_delta_t();

          double d_c = 0.0;
          double d_p = 0.0;

          if (current_time <= final_load_time)
            {
              d_c = (current_time / final_load_time) * d_max;

              if (current_time > delta_time)
                d_p = ((current_time - delta_time) / final_load_time) * d_max;
            }
          else
            {
              d_c = d_max;
              d_p = d_max;
            }

          const double r_c = std::sqrt(
            2 * r_i * std::abs(d_c) -
            std::abs(d_c) * std::abs(d_c)); // current indentation radius

          displ_incr[0] = d_c; // current indentation depth
          displ_incr[1] = d_p; // previous indentation depth
          displ_incr[2] = r_i; // indenter radius
          displ_incr[3] = r_c; // current indentation radius
        }

      return displ_incr;
    }
  };


  //@sect4{Base class: Odeometric testing device in TU Graz}
  template <int dim>
  class OdeometerGraz : public Solid<dim>
  {
  public:
    OdeometerGraz(const Parameters::AllParameters &parameters)
      : Solid<dim>(parameters)
    {}
    virtual ~OdeometerGraz()
    {}

  private:
    virtual void
    make_grid() override
    {
      const Point<dim - 1> mesh_center(0.0, 0.0);
      const double         radius = this->parameters.radius;
      const double         height = this->parameters.height;

      // Create a quarter_hyper_ball in 2d, i.e. a quarter-circle and extrude it
      // to obtain a quarter cylinder
      Triangulation<dim - 1> triangulation_in;
      GridGenerator::quarter_hyper_ball(triangulation_in, mesh_center, radius);
      GridGenerator::extrude_triangulation(triangulation_in,
                                           3,
                                           height,
                                           this->triangulation);

      // Assign a cylindrical manifold to the geometry
      const CylindricalManifold<dim> cylinder_3d(2);
      const types::manifold_id       cylinder_id = 0;
      this->triangulation.set_manifold(cylinder_id, cylinder_3d);

      // Assign proper boundary ids
      for (auto cell : this->triangulation.active_cell_iterators())
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() == true)
                {
                  if (cell->face(face)->center()[2] == 0.0)
                    cell->face(face)->set_boundary_id(1); // bottom
                  else if (cell->face(face)->center()[2] == height)
                    cell->face(face)->set_boundary_id(2);        // top
                  else if (cell->face(face)->center()[0] == 0.0) //-4.0
                    cell->face(face)->set_boundary_id(3);        // left
                  else if (cell->face(face)->center()[1] == 0.0)
                    cell->face(face)->set_boundary_id(4); // front
                  else
                    {
                      cell->face(face)->set_boundary_id(0);
                      cell->face(face)->set_all_manifold_ids(cylinder_id);
                    }
                }
            }
        }

      // Scale and refine the triangulation
      GridTools::scale(this->parameters.scale, this->triangulation);
      this->triangulation.refine_global(
        std::max(1U, this->parameters.global_refinement));
    }

    virtual void
    define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) override
    {
      tracked_vertices[0][0] = 0.0 * this->parameters.scale;
      tracked_vertices[0][1] = 0.0 * this->parameters.scale;
      tracked_vertices[0][2] = this->parameters.height * this->parameters.scale;

      tracked_vertices[1][0] = this->parameters.radius * this->parameters.scale;
      tracked_vertices[1][1] = 0.0 * this->parameters.scale;
      tracked_vertices[1][2] = this->parameters.height * this->parameters.scale;
    }

    virtual void
    make_dirichlet_constraints(AffineConstraints<double> &constraints) override
    {
      // Fluid pressure load on cylinder top
      if (this->parameters.load_type == "pressure")
        {
          const std::vector<double> value = get_dirichlet_load(2, 2);
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            2,
            ConstantFunction<dim>(value[2], this->n_components),
            constraints,
            this->fe.component_mask(this->pressure));
        }

      // Bottom drained
      if (this->parameters.bottom_drained == "drained")
        {
          if (this->time->get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                1,
                ConstantFunction<dim>(this->parameters.drained_pressure,
                                      this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
          else
            {
              VectorTools::interpolate_boundary_values(
                this->dof_handler_ref,
                1,
                ZeroFunction<dim>(this->n_components),
                constraints,
                this->fe.component_mask(this->pressure));
            }
        }

      // Cylinder bottom is fully fixed in space (glued)
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        1,
        ZeroFunction<dim>(this->n_components),
        constraints,
        (this->fe.component_mask(this->x_displacement) |
         this->fe.component_mask(this->y_displacement) |
         this->fe.component_mask(this->z_displacement)));

      // Define symmetry boundary conditions for lateral surfaces
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        3,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->x_displacement));
      VectorTools::interpolate_boundary_values(
        this->dof_handler_ref,
        4,
        ZeroFunction<dim>(this->n_components),
        constraints,
        this->fe.component_mask(this->y_displacement));

      // Cylinder hull confined
      if (this->parameters.lateral_confined == "confined")
        {
          VectorTools::interpolate_boundary_values(
            this->dof_handler_ref,
            0,
            ZeroFunction<dim>(this->n_components),
            constraints,
            (this->fe.component_mask(this->x_displacement) |
             this->fe.component_mask(this->y_displacement)));
        }
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        // AssertThrow(false, ExcMessage("Pressure loading not implemented for
        // rheometer examples."));

        (void)boundary_id;
      (void)pt;
      (void)N;
      return Tensor<1, dim>();
    }

    virtual types::boundary_id
    get_reaction_boundary_id_for_output() const override
    {
      return 1;
    }

    virtual double
    get_prescribed_fluid_flow(const types::boundary_id &boundary_id,
                              const Point<dim>         &pt) const override
    {
      (void)pt;
      (void)boundary_id;
      return 0.0;
    }

    virtual std::pair<types::boundary_id, types::boundary_id>
    get_drained_boundary_id_for_output() const override
    {
      if (this->parameters.lateral_drained == "drained" &&
          this->parameters.bottom_drained == "drained")
        {
          return std::make_pair(0, 1);
        }
      else if (this->parameters.lateral_drained == "drained")
        {
          return std::make_pair(0, 0);
        }
      else
        {
          return std::make_pair(1, 1);
        }
    }

    // Define Dirichlet load, definition in derived classes
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override = 0;
  };

  //@sect4{Derived class: Tension and Compression Relaxation}
  template <int dim>
  class OdeometerGrazConstant : public OdeometerGraz<dim>
  {
  public:
    OdeometerGrazConstant(const Parameters::AllParameters &parameters)
      : OdeometerGraz<dim>(parameters)
    {}
    virtual ~OdeometerGrazConstant()
    {}

  private:
    virtual std::vector<double>
    get_dirichlet_load(const types::boundary_id &boundary_id,
                       const int                &direction) const override
    {
      std::vector<double> displ_incr(
        dim, 0.0); // vector of length dim with zero entries
      if (this->parameters.load_type == "pressure")
        {
          if ((boundary_id == 2) && (direction == 2))
            {
              const double final_displ     = std::abs(this->parameters.load);
              const double final_load_time = this->parameters.end_load_time;
              const double current_time    = this->time->get_current();
              const double delta_time      = this->time->get_delta_t();

              double current_displ  = 0.0;
              double previous_displ = 0.0;

              if (current_time <= final_load_time)
                {
                  current_displ =
                    (current_time / final_load_time) * final_displ;

                  if (current_time > delta_time)
                    previous_displ =
                      ((current_time - delta_time) / final_load_time) *
                      final_displ;

                  displ_incr[2] = current_displ - previous_displ;
                }
              else
                displ_incr[2] = 0.0;
            }
        }
      return displ_incr;
    }

    virtual Tensor<1, dim>
    get_neumann_traction(const types::boundary_id &boundary_id,
                         const Point<dim>         &pt,
                         const Tensor<1, dim>     &N) const override
    {
      if (this->parameters.load_type == "pressure")
        {
          if (boundary_id == 2)
            {
              // return this->parameters.load * N;

              const double final_load   = this->parameters.load;
              const double final_time   = this->parameters.end_load_time;
              const double current_time = this->time->get_current();
              double       load;

              // linear increasing load
              if (current_time <= final_time)
                {
                  load = final_load * (current_time / final_time);
                }
              else
                {
                  load = final_load;
                }
              return load * N;
            }
        }
      (void)pt;
      return Tensor<1, dim>();
    }
  };

} // namespace NonLinearPoroViscoElasticity
