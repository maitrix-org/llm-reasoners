

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(ontable b)
(ontable c)
(on d a)
(clear b)
(clear c)
(clear d)
)
(:goal
(and
(on a c)
(on b a)
(on c d))
)
)


