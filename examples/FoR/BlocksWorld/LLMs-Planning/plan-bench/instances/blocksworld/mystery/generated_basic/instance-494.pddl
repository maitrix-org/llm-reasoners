

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(planet b)
(craves c b)
(planet d)
(planet e)
(province a)
(province c)
(province d)
(province e)
)
(:goal
(and
(craves d c)
(craves e a))
)
)


