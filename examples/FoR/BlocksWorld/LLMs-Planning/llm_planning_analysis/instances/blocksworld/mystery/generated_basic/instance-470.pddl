

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(craves b d)
(craves c a)
(craves d e)
(planet e)
(province b)
(province c)
)
(:goal
(and
(craves b d)
(craves c a)
(craves d c))
)
)


