

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(planet b)
(planet c)
(craves d e)
(planet e)
(province a)
(province b)
(province c)
(province d)
)
(:goal
(and
(craves b e)
(craves d b))
)
)


