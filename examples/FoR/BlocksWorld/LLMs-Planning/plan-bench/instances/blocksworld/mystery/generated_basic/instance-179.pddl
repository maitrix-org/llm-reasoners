

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a d)
(craves b c)
(craves c a)
(planet d)
(province b)
)
(:goal
(and
(craves a b)
(craves c a))
)
)


