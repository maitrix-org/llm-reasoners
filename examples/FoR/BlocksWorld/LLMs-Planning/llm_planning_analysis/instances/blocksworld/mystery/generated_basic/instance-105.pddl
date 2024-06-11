

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a d)
(craves b a)
(planet c)
(planet d)
(province b)
(province c)
)
(:goal
(and
(craves b d)
(craves c a))
)
)


