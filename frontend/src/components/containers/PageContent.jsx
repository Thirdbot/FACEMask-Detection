import React from 'react'

const PageContent = ({ children, className }) => {
  return (
    <div className={`p-8 w-4/5 ${className}`}>{children}</div>
  )
}

export default PageContent;