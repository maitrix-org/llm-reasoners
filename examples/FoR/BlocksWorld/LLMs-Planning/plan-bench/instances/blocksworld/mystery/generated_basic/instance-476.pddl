

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(craves b e)
(craves c d)
(craves d a)
(planet e)
(province b)
(province c)
)
(:goal
(and
(craves a e)
(craves c a)
(craves d b)
(craves e d))
)
)


