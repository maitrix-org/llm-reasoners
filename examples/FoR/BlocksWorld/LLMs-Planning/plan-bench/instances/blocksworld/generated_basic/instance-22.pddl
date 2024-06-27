

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(ontable b)
(on c a)
(on d c)
(clear b)
(clear d)
)
(:goal
(and
(on b c)
(on d a))
)
)


