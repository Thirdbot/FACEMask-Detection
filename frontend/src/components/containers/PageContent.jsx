import React from 'react'

const PageContent = ({ children, className }) => {
  return (
    <div className={`p-10 w-4/5 ${className} flex flex-col items-center justify-start`}>{children}</div>
  )
}

export default PageContent;